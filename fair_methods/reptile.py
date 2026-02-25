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

        self.meta_model = None
        self.group_params = {}
        self.group_models = {}
        self.datos_cargados = False
        self.input_dim = None
        self.sensitive_train = None
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_data(self, X_train, y_train, X_test):
        self.X_train = X_train.float().to(device)
        self.y_train = y_train.float().to(device)
        self.X_test = X_test.float().to(device)
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def _sample_group_batch(self, group_id, batch_size):
        idxs = np.where(self.sensitive_train == group_id)[0]
        if idxs.size == 0:
            return None, None
        replace = idxs.size < batch_size
        chosen = np.random.choice(idxs, size=batch_size, replace=replace)
        idx_tensor = torch.as_tensor(chosen, device=device, dtype=torch.long)
        return (
            self.X_train.index_select(0, idx_tensor),
            self.y_train.index_select(0, idx_tensor),
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
            chosen = np.random.choice(indices, size=self.inner_batch_size, replace=replace)
            idx_tensor = torch.as_tensor(chosen, device=device, dtype=torch.long)
            Xb = self.X_train.index_select(0, idx_tensor)
            yb = self.y_train.index_select(0, idx_tensor)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    def fit(self, sensitive_labels, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")

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
            task_ids = np.random.choice(
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
        self.group_models = {}
        for g_id in unique_groups:
            idxs = np.where(self.sensitive_train == g_id)[0]
            replace = idxs.size < self.k_support
            support_idx = np.random.choice(idxs, size=self.k_support, replace=replace)

            adapted = GenericModel(self.input_dim).to(device)
            adapted.load_state_dict(self.meta_model.state_dict())

            self._inner_train_on_indices(adapted, support_idx)

            adapted.eval()
            self.group_models[g_id] = adapted
            self.group_params[g_id] = {name: param.detach() for name, param in adapted.named_parameters()}

        print("[Reptile] Adaptation complete.")

    def predict(self, sensitive_labels=None, **kwargs):
        if self.meta_model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        n_samples = self.X_test.shape[0]
        predictions = torch.zeros(n_samples, device=device)

        if sensitive_labels is not None:
            if isinstance(sensitive_labels, torch.Tensor):
                test_groups = sensitive_labels.cpu().numpy()
            else:
                test_groups = np.asarray(sensitive_labels)

            for g_id in np.unique(test_groups):
                mask = test_groups == g_id
                mask_tensor = torch.tensor(mask, device=device)
                X_group = self.X_test[mask_tensor]
                if X_group.shape[0] == 0:
                    continue

                group_model = self.group_models.get(g_id)
                if group_model is not None:
                    with torch.no_grad():
                        logits = group_model(X_group)
                        predictions[mask_tensor] = torch.sigmoid(logits)
                elif self.group_params.get(g_id) is not None:
                    params = self.group_params[g_id]
                    with torch.no_grad():
                        logits = self.meta_model(X_group, params=params)
                        predictions[mask_tensor] = torch.sigmoid(logits)
                else:
                    with torch.no_grad():
                        logits = self.meta_model(X_group)
                        predictions[mask_tensor] = torch.sigmoid(logits)
        else:
            with torch.no_grad():
                logits = self.meta_model(self.X_test)
                predictions = torch.sigmoid(logits)

        return predictions.unsqueeze(1).cpu().numpy()
