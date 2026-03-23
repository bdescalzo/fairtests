import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .fair_method import FairMethod


device = "cuda" if torch.cuda.is_available() else "cpu"


class _MMPFNet(nn.Module):
    def __init__(self, input_dim, hidden_units=(512, 512), output_dim=2):
        super().__init__()
        h1, h2 = hidden_units
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, output_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.head(x)


class MinimaxParetoFairness(FairMethod):
    def __init__(
        self,
        lr=5e-4,
        max_epochs=1000,
        batch_size=32,
        niter=5,
        patience=10,
        lrdecay=0.25,
        alpha=0.5,
        k_ini=1,
        k_min=20,
        risk_round_factor=3,
        reset_optimizer=False,
        hidden_units=(512, 512),
        balanced_sampler=True,
        seed=42,
        n_print=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.niter = niter
        self.patience = patience
        self.lrdecay = lrdecay
        self.alpha = alpha
        self.k_ini = k_ini
        self.k_min = k_min
        self.risk_round_factor = risk_round_factor
        self.reset_optimizer = reset_optimizer
        self.hidden_units = hidden_units
        self.balanced_sampler = balanced_sampler
        self.seed = seed
        self.n_print = n_print
        self.predict_batch_size = kwargs.get("predict_batch_size", 8192)

        self.model = None
        self.optimizer = None
        self.datos_cargados = False

        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.input_dim = None
        self.mu_penalty = None
        self.group_values = None
        self.group_to_index = None
        self.best_state = None
        self.X_val_external = None
        self.y_val_external = None
        self.sensitive_val_external = None
        self.rng = None
        self.torch_generator = None

    def load_data(self, X_train, y_train, X_test):
        # Keep the full dataset on CPU and stream mini-batches to device.
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.long().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def set_validation_data(self, X_val, y_val, sensitive_val):
        self.X_val_external = X_val
        self.y_val_external = y_val
        self.sensitive_val_external = sensitive_val

    @staticmethod
    def _pareto_check(table_results, eps_pareto=0.0):
        pareto = np.zeros(table_results.shape[1], dtype=np.int32)
        if table_results.shape[1] == 1:
            pareto[0] = 1
            return pareto

        for i in range(table_results.shape[1]):
            dif = table_results[:, i][:, None] - table_results
            dif = dif < eps_pareto
            dif[:, i] = True
            dif = np.sum(dif, axis=0)
            pareto[i] = int(np.prod(dif) > 0)

        if np.sum(pareto) == 0:
            ix = int(np.argmin(np.sum(table_results, axis=0)))
            pareto[ix] = 1
        return pareto

    def _make_group_map(self, sensitive_labels):
        sensitive = np.asarray(sensitive_labels)
        self.group_values = np.unique(sensitive)
        self.group_to_index = {g: i for i, g in enumerate(self.group_values)}
        mapped = np.array([self.group_to_index[g] for g in sensitive], dtype=np.int64)
        return mapped

    def _encode_existing_groups(self, sensitive_labels):
        sensitive = np.asarray(sensitive_labels)
        idx = np.array([self.group_to_index.get(g, -1) for g in sensitive], dtype=np.int64)
        valid = idx >= 0
        return idx, valid

    @staticmethod
    def _to_one_hot_binary(y):
        y = y.long()
        y = torch.clamp(y, min=0, max=1)
        return F.one_hot(y, num_classes=2).float()

    def _build_loader(self, X, y, g_idx, train=False):
        y_long = y if y.dtype == torch.long else y.long()
        dataset = TensorDataset(
            X,
            y_long,
            torch.as_tensor(g_idx, dtype=torch.long),
        )
        if train and self.balanced_sampler:
            counts = np.bincount(g_idx, minlength=len(self.group_values)).astype(np.float64)
            counts[counts == 0] = 1.0
            sample_w = 1.0 / counts[g_idx]
            sample_w = torch.as_tensor(sample_w, dtype=torch.double)
            sampler = WeightedRandomSampler(
                sample_w,
                len(sample_w),
                replacement=True,
                generator=self.torch_generator,
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=(device == "cuda"),
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            pin_memory=(device == "cuda"),
            generator=self.torch_generator,
        )

    def _epoch_linearweight(self, loader, train_type):
        n_groups = len(self.group_values)
        base_loss_sum = np.zeros(n_groups, dtype=np.float64)
        acc_sum = np.zeros(n_groups, dtype=np.float64)
        count_sum = np.zeros(n_groups, dtype=np.float64)

        train = train_type == "Train"
        self.model.train(mode=train)

        pin_memory = device == "cuda"
        for x, y, sensitive_idx in loader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            sensitive_idx = sensitive_idx.to(device, non_blocking=pin_memory)
            utility_1hot = self._to_one_hot_binary(y)
            if train:
                self.optimizer.zero_grad()

            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            base_loss = -(log_probs * utility_1hot).sum(dim=-1)

            sens_1hot = F.one_hot(sensitive_idx, num_classes=n_groups).float()
            group_denom = sens_1hot.sum(dim=0).clamp(min=1.0)
            group_loss = (base_loss.unsqueeze(-1) * sens_1hot).sum(dim=0) / group_denom
            full_loss = (group_loss * self.mu_penalty).sum()

            if train:
                full_loss.backward()
                self.optimizer.step()

            pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
            gt = utility_1hot.argmax(dim=-1)
            acc = (pred == gt).float()

            for g in torch.unique(sensitive_idx):
                gi = int(g.item())
                mask = sensitive_idx == g
                c = int(mask.sum().item())
                if c == 0:
                    continue
                base_loss_sum[gi] += float(base_loss[mask].sum().item())
                acc_sum[gi] += float(acc[mask].sum().item())
                count_sum[gi] += c

        base_loss_out = np.zeros(n_groups, dtype=np.float64)
        acc_out = np.zeros(n_groups, dtype=np.float64)
        for g in range(n_groups):
            if count_sum[g] > 0:
                base_loss_out[g] = base_loss_sum[g] / count_sum[g]
                acc_out[g] = acc_sum[g] / count_sum[g]

        full_loss_out = float(np.sum(base_loss_out * self.mu_penalty.detach().cpu().numpy()))
        return base_loss_out, acc_out, full_loss_out

    def _adaptive_optimizer(self, train_loader, val_loader):
        best_val_full = None
        best_base_loss_val = None
        best_model_state = None
        best_opt_state = None
        best_epoch = 0

        patience_counter = 0
        local_lrdecay = self.lrdecay

        for epoch in range(self.max_epochs):
            _, _, train_full = self._epoch_linearweight(train_loader, train_type="Train")
            val_base_loss, val_acc, val_full = self._epoch_linearweight(
                val_loader, train_type="Val"
            )

            if best_val_full is None:
                best_val_full = val_full
                best_base_loss_val = val_base_loss.copy()
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_opt_state = copy.deepcopy(self.optimizer.state_dict())
                best_epoch = epoch + 1
            else:
                if val_full < 0.999 * best_val_full:
                    best_val_full = val_full
                    best_base_loss_val = val_base_loss.copy()
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_opt_state = copy.deepcopy(self.optimizer.state_dict())
                    patience_counter = 0
                    local_lrdecay = self.lrdecay
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1
                    if best_val_full < val_full and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        self.optimizer.load_state_dict(best_opt_state)
                        self.optimizer.param_groups[0]["lr"] *= local_lrdecay
                        local_lrdecay *= self.lrdecay

            if ((epoch % self.n_print == self.n_print - 1) and (epoch >= 1)) or epoch == 0:
                print(
                    f"[MMPF][Adaptive] epoch={epoch} lr={self.optimizer.param_groups[0]['lr']:.2e} "
                    f"loss_tr={train_full:.4f} loss_val={val_full:.4f} "
                    f"base_val={np.round(val_base_loss, 4)} acc_val={np.round(val_acc, 4)} "
                    f"stop_c={patience_counter}"
                )

            if patience_counter > self.patience:
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.optimizer.load_state_dict(best_opt_state)

        return best_base_loss_val, best_epoch

    def fit(self, sensitive_labels=None, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")
        if sensitive_labels is None:
            raise ValueError("Se requieren etiquetas sensibles para entrenar MMPF")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.torch_generator = torch.Generator().manual_seed(self.seed)

        s_train = (
            sensitive_labels.detach().cpu().numpy()
            if isinstance(sensitive_labels, torch.Tensor)
            else np.asarray(sensitive_labels)
        )
        g_train_idx = self._make_group_map(s_train)

        X_val = self.X_val_external
        y_val = self.y_val_external
        s_val = self.sensitive_val_external
        val_fraction = float(kwargs.get("val_fraction", 0.2))

        if X_val is None or y_val is None or s_val is None:
            n = self.X_train.shape[0]
            perm = self.rng.permutation(n)
            n_val = max(1, int(round(val_fraction * n)))
            val_ids = perm[:n_val]
            tr_ids = perm[n_val:]

            X_fit = self.X_train[tr_ids]
            y_fit = self.y_train[tr_ids]
            g_fit = g_train_idx[tr_ids]

            X_val_t = self.X_train[val_ids]
            y_val_t = self.y_train[val_ids]
            g_val = g_train_idx[val_ids]
        else:
            X_fit = self.X_train
            y_fit = self.y_train
            g_fit = g_train_idx

            X_val_t = X_val.float().cpu()
            y_val_t = y_val.long().cpu()

            s_val_np = s_val.detach().cpu().numpy() if isinstance(s_val, torch.Tensor) else np.asarray(s_val)
            g_val, valid = self._encode_existing_groups(s_val_np)
            if not np.all(valid):
                keep = torch.as_tensor(valid, dtype=torch.bool)
                X_val_t = X_val_t[keep]
                y_val_t = y_val_t[keep]
                g_val = g_val[valid]

        if X_fit.shape[0] == 0 or X_val_t.shape[0] == 0:
            raise ValueError("No hay suficientes datos tras construir train/val para MMPF")

        train_loader = self._build_loader(X_fit, y_fit, g_fit, train=True)
        val_loader = self._build_loader(X_val_t, y_val_t, g_val, train=False)

        self.model = _MMPFNet(self.input_dim, hidden_units=self.hidden_units).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        n_groups = len(self.group_values)
        mu_i = np.ones(n_groups, dtype=np.float64)
        mu_i /= mu_i.sum()

        risk_list = []
        K = int(self.k_ini)
        i_patience = 0

        risk_max_best = None
        best_mu = mu_i.copy()
        best_state = copy.deepcopy(self.model.state_dict())

        iteration = 0
        while iteration <= self.niter and i_patience <= self.patience:
            self.mu_penalty = torch.as_tensor(mu_i / mu_i.sum(), dtype=torch.float32, device=device)
            print(f"[MMPF] Iteration {iteration} | mu={np.round(mu_i, 4)}")

            if self.reset_optimizer:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            best_base_loss_val, _ = self._adaptive_optimizer(train_loader, val_loader)
            risk = np.round(best_base_loss_val.astype(np.float64), self.risk_round_factor)
            risk_max = float(np.max(risk))
            argrisk_max = np.where(risk == risk_max)[0]

            if risk_max_best is None:
                risk_max_best = risk_max + 1.0

            improved = risk_max_best > risk_max
            if improved:
                risk_max_best = risk_max
                best_mu = mu_i.copy()
                best_state = copy.deepcopy(self.model.state_dict())
                K = min(K, self.k_min)
                i_patience = 0
                step_type = "improved"
            else:
                K += 1
                i_patience += 1
                step_type = "no_improve"

            self.model.load_state_dict(best_state)

            step_mask = (risk >= risk_max_best).astype(np.float64)
            step_mu = step_mask / step_mask.sum()

            print(
                f"[MMPF] Iter {iteration} ({step_type}) risk_max={risk_max:.4f} "
                f"best_max={risk_max_best:.4f} argmax={argrisk_max.tolist()} K={K}"
            )

            risk_list.append(risk.copy())
            mu_i = (1.0 - self.alpha) * mu_i + (self.alpha / max(1.0, float(K))) * step_mu
            mu_i = mu_i / mu_i.sum()

            risk_np = np.array(risk_list, dtype=np.float64)
            pareto_mask = self._pareto_check(risk_np.transpose())
            if pareto_mask[-1] == 0:
                self.optimizer.param_groups[0]["lr"] *= self.lrdecay
                print(f"[MMPF] Pareto-dominated step. Decayed lr to {self.optimizer.param_groups[0]['lr']:.2e}")

            iteration += 1

        self.model.load_state_dict(best_state)
        self.best_state = copy.deepcopy(best_state)
        self.mu_penalty = torch.as_tensor(best_mu, dtype=torch.float32, device=device)

    def predict(self, sensitive_labels=None, **kwargs):
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        predict_batch_size = int(kwargs.get("batch_size", self.predict_batch_size))
        loader = DataLoader(
            TensorDataset(self.X_test),
            batch_size=predict_batch_size,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

        self.model.eval()
        chunks = []
        with torch.no_grad():
            for (Xb,) in loader:
                Xb = Xb.to(device, non_blocking=(device == "cuda"))
                logits = self.model(Xb)
                chunks.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        probs = torch.cat(chunks, dim=0)
        return probs.unsqueeze(1).numpy()


# Backwards compatibility alias
MinimaxParetoFair = MinimaxParetoFairness
