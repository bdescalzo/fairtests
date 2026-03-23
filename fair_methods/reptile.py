import numpy as np
import torch
from torch import nn

from .fair_method import FairMethod
from .models import GenericModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class Reptile(FairMethod):
    def __init__(
        self,
        inner_lr=0.005,
        inner_steps=10,
        meta_epochs=200,
        meta_lr=0.003,
        inner_batch_size=128,
        k_support=128,
        meta_batch_size=4,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.inner_batch_size = inner_batch_size
        self.k_support = k_support
        self.meta_batch_size = meta_batch_size
        self.seed = seed

        self.meta_model = None
        self.group_params = {}
        self.datos_cargados = False
        self.input_dim = None
        self.sensitive_train = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.predict_batch_size = kwargs.get("predict_batch_size", 8192)
        self.rng = None

    def load_data(self, X_train, y_train, X_test):
        # Keep full datasets on CPU and stream mini-batches to the device.
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.float().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def _sample_group_batch(self, group_id, batch_size):
        idxs = np.where(self.sensitive_train == group_id)[0]
        if idxs.size == 0:
            return None, None
        replace = idxs.size < batch_size
        chosen = self.rng.choice(idxs, size=batch_size, replace=replace)
        idx_tensor = torch.as_tensor(chosen, dtype=torch.long)
        return (
            self.X_train.index_select(0, idx_tensor).to(device),
            self.y_train.index_select(0, idx_tensor).to(device),
        )

    def _inner_train(self, model, group_id):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.inner_lr)
        model.train()
        for _ in range(self.inner_steps):
            Xb, yb = self._sample_group_batch(group_id, self.inner_batch_size)
            if Xb is None:
                continue
            optimizer.zero_grad()
            logits = model(Xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    def _inner_train_on_indices(self, model, indices):
        if indices.size == 0:
            return
        optimizer = torch.optim.Adam(model.parameters(), lr=self.inner_lr)
        model.train()
        for _ in range(self.inner_steps):
            replace = indices.size < self.inner_batch_size
            chosen = self.rng.choice(
                indices, size=self.inner_batch_size, replace=replace
            )
            idx_tensor = torch.as_tensor(chosen, dtype=torch.long)
            Xb = self.X_train.index_select(0, idx_tensor).to(device)
            yb = self.y_train.index_select(0, idx_tensor).to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    def fit(self, sensitive_labels, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)

        # Update loss to handle class imbalance in the training data
        with torch.no_grad():
            y = self.y_train
            pos = torch.sum(y == 1).float()
            neg = torch.sum(y == 0).float()
            if pos > 0:
                pos_weight = (neg / pos).clamp(min=1.0)
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if isinstance(sensitive_labels, torch.Tensor):
            self.sensitive_train = sensitive_labels.cpu().numpy()
        else:
            self.sensitive_train = np.asarray(sensitive_labels)

        unique_groups = np.unique(self.sensitive_train)
        if unique_groups.size == 0:
            raise ValueError("No hay grupos sensibles para entrenar Reptile")

        self.meta_model = GenericModel(self.input_dim).to(device)

        print(
            f"[Reptile] Meta-training on {unique_groups.size} groups "
            f"(epochs={self.meta_epochs}, inner_steps={self.inner_steps}, "
            f"inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"inner_batch_size={self.inner_batch_size})"
        )

        for epoch in range(self.meta_epochs):
            task_ids = self.rng.choice(
                unique_groups, size=self.meta_batch_size, replace=True
            )
            deltas = [torch.zeros_like(p) for p in self.meta_model.parameters()]

            for group_id in task_ids:
                # Clone meta-model
                adapted = GenericModel(self.input_dim).to(device)
                adapted.load_state_dict(self.meta_model.state_dict())

                # Inner-loop training on task
                self._inner_train(adapted, group_id)

                # Accumulate meta-update direction
                with torch.no_grad():
                    for i, (meta_param, adapted_param) in enumerate(
                        zip(self.meta_model.parameters(), adapted.parameters())
                    ):
                        deltas[i] += (adapted_param.data - meta_param.data)

            # Meta-update: move initialization toward adapted parameters
            with torch.no_grad():
                scale = self.meta_lr / float(self.meta_batch_size)
                for meta_param, delta in zip(self.meta_model.parameters(), deltas):
                    meta_param.data += scale * delta

            if (epoch + 1) % 20 == 0:
                print(f"[Reptile] Meta-epoch {epoch + 1}/{self.meta_epochs}")

        # Adaptation for inference (K-shot from training data)
        print("[Reptile] Adapting per-group parameters...")
        self.group_params = {}
        for g_id in unique_groups:
            idxs = np.where(self.sensitive_train == g_id)[0]
            replace = idxs.size < self.k_support
            support_idx = self.rng.choice(idxs, size=self.k_support, replace=replace)

            adapted = GenericModel(self.input_dim).to(device)
            adapted.load_state_dict(self.meta_model.state_dict())

            self._inner_train_on_indices(adapted, support_idx)

            self.group_params[g_id] = {
                name: param.detach().cpu() for name, param in adapted.named_parameters()
            }

        print("[Reptile] Adaptation complete.")

    def predict(self, sensitive_labels=None, **kwargs):
        if self.meta_model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        n_samples = self.X_test.shape[0]
        predictions = torch.zeros(n_samples, dtype=torch.float32)
        batch_size = int(kwargs.get("batch_size", self.predict_batch_size))

        if sensitive_labels is not None:
            if isinstance(sensitive_labels, torch.Tensor):
                test_groups = sensitive_labels.cpu().numpy()
            else:
                test_groups = np.asarray(sensitive_labels)

            for g_id in np.unique(test_groups):
                group_indices = np.where(test_groups == g_id)[0]
                if group_indices.size == 0:
                    continue

                params_cpu = self.group_params.get(g_id)
                params = None
                if params_cpu is not None:
                    params = {name: p.to(device) for name, p in params_cpu.items()}

                with torch.no_grad():
                    for start in range(0, group_indices.size, batch_size):
                        end = min(start + batch_size, group_indices.size)
                        batch_idx = group_indices[start:end]
                        batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
                        X_group = self.X_test.index_select(0, batch_idx_t).to(device)
                        if params is not None:
                            logits = self.meta_model(X_group, params=params)
                        else:
                            logits = self.meta_model(X_group)
                        predictions[batch_idx_t] = torch.sigmoid(logits).cpu()
        else:
            with torch.no_grad():
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    idx = np.arange(start, end)
                    idx_t = torch.as_tensor(idx, dtype=torch.long)
                    X_batch = self.X_test.index_select(0, idx_t).to(device)
                    logits = self.meta_model(X_batch)
                    predictions[idx_t] = torch.sigmoid(logits).cpu()

        return predictions.unsqueeze(1).numpy()
