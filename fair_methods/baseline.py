import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fair_method import FairMethod
from .models import GenericModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class Baseline(FairMethod):
    def __init__(
        self,
        lr=1e-3,
        epochs=15,
        batch_size=1024,
        seed=42,
        model_class=None,
        **kwargs,
    ):
        super().__init__(model_class=model_class, **kwargs)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        if self.model_class is None:
            self.model_class = GenericModel
        self.model = None
        self.datos_cargados = False
        self.input_dim = None
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_data(self, X_train, y_train, X_test):
        # Keep full tensors on CPU and stream mini-batches to device.
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.float().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def fit(self, sensitive_labels=None, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = self.model_class(self.input_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        data_generator = torch.Generator().manual_seed(self.seed)

        dataset = TensorDataset(self.X_train, self.y_train)
        pin_memory = device == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            generator=data_generator,
        )

        print(
            f"[Baseline] Training for {self.epochs} epochs "
            f"(batch_size={self.batch_size}, lr={self.lr})"
        )
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total = 0
            for Xb, yb in loader:
                Xb = Xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * Xb.size(0)
                total += Xb.size(0)
            if (epoch + 1) % 5 == 0:
                print(
                    f"[Baseline] Epoch {epoch + 1}/{self.epochs} | "
                    f"loss={total_loss / max(1, total):.4f}"
                )

    def predict(self, **kwargs):
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        predict_batch_size = int(kwargs.get("batch_size", 8192))
        pin_memory = device == "cuda"
        loader = DataLoader(
            TensorDataset(self.X_test),
            batch_size=predict_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

        self.model.eval()
        chunks = []
        with torch.no_grad():
            for (Xb,) in loader:
                Xb = Xb.to(device, non_blocking=pin_memory)
                logits = self.model(Xb)
                chunks.append(torch.sigmoid(logits).cpu())
        probs = torch.cat(chunks, dim=0)
        return probs.unsqueeze(1).numpy()
