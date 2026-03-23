import numpy as np
import torch
from torch import nn

from .fair_method import FairMethod
from .models import GenericModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetaLearning(FairMethod):
    def __init__(
        self,
        inner_lr=0.005,
        inner_steps=5,
        meta_epochs=150,
        meta_lr=3e-4,
        k_support=128,
        k_query=128,
        replace=False,
        group_budget_divisor=4,
        support_fraction=1 / 3,
        use_full_data=False,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.k_support = k_support
        self.k_query = k_query
        self.replace = replace
        self.group_budget_divisor = group_budget_divisor
        self.support_fraction = support_fraction
        self.use_full_data = use_full_data
        self.seed = seed

        self.meta_model = None
        self.group_params = {}
        self.datos_cargados = False
        self.input_dim = None
        self.sensitive_train = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.effective_k_support = k_support
        self.effective_k_query = k_query
        self.effective_task_budget = k_support + k_query
        self.predict_batch_size = kwargs.get("predict_batch_size", 8192)
        self.rng = None

    def load_data(self, X_train, y_train, X_test):
        # Keep datasets on CPU to avoid duplicating large tensors in VRAM.
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.float().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def _sample_task_batches(self, group_id):
        idxs = np.where(self.sensitive_train == group_id)[0]
        k_support = self.effective_k_support
        k_query = self.effective_k_query
        if self.replace:
            replace = idxs.size < (k_support + k_query)
        else:
            replace = False
        chosen = self.rng.choice(idxs, size=k_support + k_query, replace=replace)
        support_idx = chosen[:k_support]
        query_idx = chosen[k_support:]

        support_idx_t = torch.as_tensor(support_idx, dtype=torch.long)
        query_idx_t = torch.as_tensor(query_idx, dtype=torch.long)

        support_x = self.X_train.index_select(0, support_idx_t).to(device)
        support_y = self.y_train.index_select(0, support_idx_t).to(device)
        query_x = self.X_train.index_select(0, query_idx_t).to(device)
        query_y = self.y_train.index_select(0, query_idx_t).to(device)
        return support_x, support_y, query_x, query_y

    def _resolve_effective_k(self, unique_groups):
        self.effective_k_support = int(self.k_support)
        self.effective_k_query = int(self.k_query)
        self.effective_task_budget = self.effective_k_support + self.effective_k_query

        if self.replace:
            return

        counts = [int(np.sum(self.sensitive_train == g_id)) for g_id in unique_groups]
        min_group_size = int(np.min(counts))
        if min_group_size < 2:
            raise ValueError(
                "replace=False requires at least 2 samples in every sensitive group "
                "(one for support and one for query)."
            )

        if self.group_budget_divisor <= 0:
            raise ValueError("group_budget_divisor must be > 0")
        if not (0.0 < self.support_fraction < 1.0):
            raise ValueError("support_fraction must be strictly between 0 and 1")

        # Tie the per-group task budget to the smallest group so every group can
        # sustain the same support/query episode size without near-full reuse.
        effective_total = max(2, min_group_size // int(self.group_budget_divisor))
        effective_k_support = int(round(effective_total * float(self.support_fraction)))
        effective_k_support = max(1, min(effective_k_support, effective_total - 1))
        effective_k_query = effective_total - effective_k_support

        self.effective_task_budget = effective_total
        self.effective_k_support = effective_k_support
        self.effective_k_query = effective_k_query

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
            raise ValueError("No hay grupos sensibles para entrenar MAML")

        self._resolve_effective_k(unique_groups)
        self.meta_model = GenericModel(self.input_dim).to(device)
        meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.meta_lr)

        print(
            f"[MAML] Meta-training on {unique_groups.size} groups "
            f"(epochs={self.meta_epochs}, inner_steps={self.inner_steps}, "
            f"inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"N={self.effective_task_budget}, "
            f"k_support={self.effective_k_support}, k_query={self.effective_k_query}, "
            f"replace={self.replace})"
        )
        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            meta_loss = 0.0
            valid_tasks = 0

            for g_id in unique_groups:
                support_x, support_y, query_x, query_y = self._sample_task_batches(g_id)
                if support_x is None:
                    continue

                params = {name: param for name, param in self.meta_model.named_parameters()}

                for _ in range(self.inner_steps):
                    support_logits = self.meta_model(support_x, params)
                    support_loss = self.loss_fn(support_logits, support_y)
                    grads = torch.autograd.grad(
                        support_loss, params.values(), create_graph=True
                    )
                    params = {
                        name: param - self.inner_lr * grad
                        for (name, param), grad in zip(params.items(), grads)
                    }

                query_logits = self.meta_model(query_x, params)
                task_loss = self.loss_fn(query_logits, query_y)
                meta_loss = meta_loss + task_loss
                valid_tasks += 1

            if valid_tasks == 0:
                continue

            meta_loss = meta_loss / valid_tasks
            meta_loss.backward()
            meta_optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Meta-epoch {epoch + 1}/{self.meta_epochs} | "
                    f"meta-loss={meta_loss.item():.4f}"
                )

        print("[MAML] Adapting per-group parameters...")
        self.group_params = {}
        for g_id in unique_groups:
            idxs = np.where(self.sensitive_train == g_id)[0]
            if self.use_full_data:
                support_idx = idxs
            else:
                if self.replace:
                    replace = idxs.size < self.effective_k_support
                else:
                    replace = False
                support_idx = self.rng.choice(
                    idxs, size=self.effective_k_support, replace=replace
                )
            support_idx_t = torch.as_tensor(support_idx, dtype=torch.long)
            X_support = self.X_train.index_select(0, support_idx_t).to(device)
            y_support = self.y_train.index_select(0, support_idx_t).to(device)
            adapted_params = {
                name: param.clone().detach().requires_grad_(True)
                for name, param in self.meta_model.named_parameters()
            }

            for _ in range(self.inner_steps):
                logits = self.meta_model(X_support, adapted_params)
                loss = self.loss_fn(logits, y_support)
                grads = torch.autograd.grad(loss, adapted_params.values())
                adapted_params = {
                    name: param - self.inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            # Persist adapted params on CPU to reduce persistent VRAM usage.
            self.group_params[g_id] = {
                name: param.detach().cpu() for name, param in adapted_params.items()
            }
        print("[MAML] Adaptation complete.")

    def predict(self, sensitive_labels=None, **kwargs):
        if self.meta_model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        self.meta_model.eval()
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
                        logits = self.meta_model(X_group, params=params)
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
