import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fair_method import FairMethod
from .meta import ModeloEnBruto

device = "cuda" if torch.cuda.is_available() else "cpu"


class Baseline(FairMethod):
    def __init__(self, lr=1e-3, epochs=15, batch_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.datos_cargados = False
        self.input_dim = None
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_data(self, X_train, y_train, X_test):
        self.X_train = X_train.float().to(device)
        self.y_train = y_train.float().to(device)
        self.X_test = X_test.float().to(device)
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def fit(self, sensitive_labels=None, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")

        self.model = ModeloEnBruto(self.input_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(
            f"[Baseline] Training for {self.epochs} epochs "
            f"(batch_size={self.batch_size}, lr={self.lr})"
        )
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total = 0
            for Xb, yb in loader:
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

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test)
            probs = torch.sigmoid(logits)
        return probs.unsqueeze(1).cpu().numpy()
