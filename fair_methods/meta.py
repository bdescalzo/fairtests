import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fair_method import FairMethod

device = "cuda" if torch.cuda.is_available() else "cpu"


# Original model (kept for reference)
# class ModeloEnBruto(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
#
#     def forward(self, x, params=None):
#         if params is None:
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#         else:
#             x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
#             x = F.relu(x)
#             x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
#             x = F.relu(x)
#             x = F.linear(x, params["fc3.weight"], params["fc3.bias"])
#         return x.squeeze(-1)


class ModeloEnBruto(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
       # self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
       # self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, params=None):
        if params is None:
            x = self.fc1(x)
        #    x = self.ln1(x)
            x = F.gelu(x)
            #x = self.dropout(x)
            x = self.fc2(x)
        #    x = self.ln2(x)
            x = F.gelu(x)
            #x = self.dropout(x)
            x = self.fc3(x)
        else:
            x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
          #  x = F.layer_norm(x, (128,), params["ln1.weight"], params["ln1.bias"])
            x = F.gelu(x)
            #x = F.dropout(x, p=0.2, training=self.training)
            x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
           # x = F.layer_norm(x, (128,), params["ln2.weight"], params["ln2.bias"])
            x = F.gelu(x)
          #  x = F.dropout(x, p=0.2, training=self.training)
            x = F.linear(x, params["fc3.weight"], params["fc3.bias"])
        return x.squeeze(-1)


class MetaLearning(FairMethod):
    def __init__(
        self,
        inner_lr=0.005,
        inner_steps=5,
        meta_epochs=150,
        meta_lr=3e-4,
        k_support=128,
        k_query=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        self.k_support = k_support
        self.k_query = k_query

        self.meta_model = None
        self.group_params = {}
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

    def _sample_task_batches(self, group_id):
        idxs = np.where(self.sensitive_train == group_id)[0]
        k_support = self.k_support
        k_query = self.k_query
        replace = idxs.size < (k_support + k_query)
        chosen = np.random.choice(idxs, size=k_support + k_query, replace=replace)
        support_idx = chosen[:k_support]
        query_idx = chosen[k_support:]

        support_x = self.X_train[support_idx]
        support_y = self.y_train[support_idx]
        query_x = self.X_train[query_idx]
        query_y = self.y_train[query_idx]
        return support_x, support_y, query_x, query_y

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
            raise ValueError("No hay grupos sensibles para entrenar MAML")

        self.meta_model = ModeloEnBruto(self.input_dim).to(device)
        meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.meta_lr)

        print(
            f"[MAML] Meta-training on {unique_groups.size} groups "
            f"(epochs={self.meta_epochs}, inner_steps={self.inner_steps}, "
            f"inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"k_support={self.k_support}, k_query={self.k_query})"
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
            replace = idxs.size < self.k_support
            support_idx = np.random.choice(idxs, size=self.k_support, replace=replace)
            X_support = self.X_train[support_idx]
            y_support = self.y_train[support_idx]
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

            self.group_params[g_id] = adapted_params
        print("[MAML] Adaptation complete.")

    def predict(self, sensitive_labels=None, **kwargs):
        if self.meta_model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        self.meta_model.eval()
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

                params = self.group_params.get(g_id)
                with torch.no_grad():
                    logits = self.meta_model(X_group, params=params)
                    predictions[mask_tensor] = torch.sigmoid(logits)
        else:
            with torch.no_grad():
                logits = self.meta_model(self.X_test)
                predictions = torch.sigmoid(logits)

        return predictions.unsqueeze(1).cpu().numpy()
