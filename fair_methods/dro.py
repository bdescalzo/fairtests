import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .fair_method import FairMethod
from .models import GenericModel


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _parse_generalization_adjustment(adjustment, n_groups):
    if adjustment is None:
        values = np.zeros(1, dtype=np.float64)
    elif isinstance(adjustment, str):
        stripped = adjustment.strip()
        if not stripped:
            values = np.zeros(1, dtype=np.float64)
        else:
            values = np.asarray(
                [float(part.strip()) for part in stripped.split(",")],
                dtype=np.float64,
            )
    elif np.isscalar(adjustment):
        values = np.asarray([float(adjustment)], dtype=np.float64)
    else:
        values = np.asarray(list(adjustment), dtype=np.float64)

    if values.size == 1:
        return np.repeat(values, n_groups)
    if values.size != n_groups:
        raise ValueError(
            "generalization_adjustment must be a scalar or provide one value per group."
        )
    return values


def _binary_logits(logits):
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(
        "Expected binary logits to have shape [N] or [N, 1]. "
        f"Received shape {tuple(logits.shape)}."
    )


def _positive_class_probs(logits):
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return torch.sigmoid(_binary_logits(logits))
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _per_sample_loss(logits, y):
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return F.binary_cross_entropy_with_logits(
            _binary_logits(logits),
            y.float(),
            reduction="none",
        )
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return F.cross_entropy(logits, y.long(), reduction="none")
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _per_sample_accuracy(logits, y):
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        preds = (_positive_class_probs(logits) >= 0.5).long()
        return (preds == y.long()).float()
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return (torch.argmax(logits, dim=-1) == y.long()).float()
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _groupwise_train_val_split(group_idx, val_fraction, rng):
    group_idx = np.asarray(group_idx, dtype=np.int64)
    unique_groups = np.unique(group_idx)

    train_parts = []
    val_parts = []
    for group_id in unique_groups:
        indices = np.flatnonzero(group_idx == group_id)
        shuffled = rng.permutation(indices)

        if shuffled.size <= 1:
            train_parts.append(shuffled)
            continue

        n_val = int(round(val_fraction * shuffled.size))
        n_val = min(max(n_val, 1), shuffled.size - 1)
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])

    train_idx = (
        np.concatenate(train_parts).astype(np.int64, copy=False)
        if train_parts
        else np.empty(0, dtype=np.int64)
    )
    val_idx = (
        np.concatenate(val_parts).astype(np.int64, copy=False)
        if val_parts
        else np.empty(0, dtype=np.int64)
    )

    if train_idx.size > 0 and val_idx.size > 0:
        return train_idx, val_idx

    n_samples = group_idx.shape[0]
    if n_samples < 2:
        raise ValueError("Se requieren al menos 2 muestras para construir train/val.")

    perm = rng.permutation(n_samples)
    n_val = min(max(int(round(val_fraction * n_samples)), 1), n_samples - 1)
    return perm[n_val:], perm[:n_val]


class _GroupDatasetInfo:
    def __init__(self, group_values, encoded_groups):
        self.group_values = np.asarray(group_values)
        self.encoded_groups = np.asarray(encoded_groups, dtype=np.int64)
        self.n_groups = int(self.group_values.shape[0])
        self._group_counts = np.bincount(
            self.encoded_groups, minlength=self.n_groups
        ).astype(np.float32, copy=False)

    def group_counts(self):
        return torch.as_tensor(self._group_counts, dtype=torch.float32)

    def group_str(self, group_idx):
        return f"group={self.group_values[int(group_idx)]}"


class _LossComputer:
    # Mirrors docs/source_implementations/group_DRO-master/loss.py::LossComputer,
    # but keeps tensors on the active device instead of hard-coding CUDA.
    def __init__(
        self,
        dataset_info,
        alpha,
        gamma=0.1,
        adj=None,
        min_var_weight=0.0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.min_var_weight = float(min_var_weight)
        self.step_size = float(step_size)
        self.normalize_loss = bool(normalize_loss)
        self.btl = bool(btl)

        self.n_groups = dataset_info.n_groups
        self.group_counts = dataset_info.group_counts().to(device)
        self.group_count_sqrt = torch.sqrt(self.group_counts.clamp(min=1.0))
        self.group_frac = self.group_counts / self.group_counts.sum().clamp(min=1.0)
        self.group_str = dataset_info.group_str

        if adj is not None:
            self.adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
        else:
            self.adj = torch.zeros(self.n_groups, dtype=torch.float32, device=device)

        self.adv_probs = torch.ones(self.n_groups, dtype=torch.float32, device=device)
        self.adv_probs = self.adv_probs / float(self.n_groups)
        self.exp_avg_loss = torch.zeros(
            self.n_groups, dtype=torch.float32, device=device
        )
        self.exp_avg_initialized = torch.zeros(
            self.n_groups, dtype=torch.bool, device=device
        )

        self.reset_stats()

    def loss(self, yhat, y, group_idx):
        per_sample_losses = _per_sample_loss(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            _per_sample_accuracy(yhat, y), group_idx
        )

        self.update_exp_avg_loss(group_loss, group_count)
        actual_loss, weights = self.compute_robust_loss(group_loss)
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss

    def compute_robust_loss(self, group_loss):
        if self.btl:
            adjusted_loss = self.exp_avg_loss + self.adj / self.group_count_sqrt
            return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss = adjusted_loss + self.adj / self.group_count_sqrt
        if self.normalize_loss:
            denom = adjusted_loss.sum().clamp(min=torch.finfo(adjusted_loss.dtype).eps)
            adjusted_loss = adjusted_loss / denom

        self.adv_probs = self.adv_probs * torch.exp(
            self.step_size * adjusted_loss.detach()
        )
        self.adv_probs = self.adv_probs / self.adv_probs.sum().clamp(min=1e-12)
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = torch.argsort(ref_loss, descending=True)
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / max(self.alpha, 1e-12)
        last_idx = int(mask.sum().item())
        if last_idx < self.n_groups:
            weights[last_idx] = 1.0 - weights.sum()
        weights = (
            sorted_frac * self.min_var_weight + weights * (1.0 - self.min_var_weight)
        )

        robust_loss = sorted_loss @ weights
        _, unsort_idx = torch.sort(sorted_idx)
        return robust_loss, weights[unsort_idx]

    def compute_group_avg(self, values, group_idx):
        groups = torch.arange(self.n_groups, device=device).unsqueeze(1)
        group_map = (group_idx == groups).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_mean = (group_map @ values.view(-1)) / group_denom
        return group_mean, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        observed = group_count > 0
        prev_weights = (
            (1.0 - self.gamma * observed.float()) * self.exp_avg_initialized.float()
        )
        curr_weights = 1.0 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = self.exp_avg_initialized | observed

    def reset_stats(self):
        zeros = torch.zeros(self.n_groups, dtype=torch.float32, device=device)
        self.processed_data_counts = zeros.clone()
        self.update_data_counts = zeros.clone()
        self.update_batch_counts = zeros.clone()
        self.avg_group_loss = zeros.clone()
        self.avg_group_acc = zeros.clone()
        self.avg_per_sample_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.avg_actual_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.avg_acc = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.batch_count = 0

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights):
        denom = self.processed_data_counts + group_count
        denom = denom + (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom

        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        batch_denom = float(self.batch_count + 1)
        self.avg_actual_loss = (
            (float(self.batch_count) / batch_denom) * self.avg_actual_loss
            + (1.0 / batch_denom) * actual_loss.detach()
        )

        self.processed_data_counts += group_count
        self.update_data_counts += group_count * (weights > 0).float()
        self.update_batch_counts += ((group_count * weights) > 0).float()
        self.batch_count += 1

        group_frac = self.processed_data_counts / self.processed_data_counts.sum().clamp(
            min=1.0
        )
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


class GroupDRO(FairMethod):
    def __init__(
        self,
        n_epochs=100,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        alpha=0.2,
        gamma=0.1,
        robust_step_size=0.01,
        generalization_adjustment=0.0,
        automatic_adjustment=False,
        use_normalized_loss=False,
        btl=False,
        minimum_variational_weight=0.0,
        reweight_groups=True,
        scheduler=False,
        val_fraction=0.2,
        momentum=0.9,
        seed=42,
        n_print=10,
        model_class=None,
        **kwargs,
    ):
        super().__init__(model_class=model_class, **kwargs)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.robust_step_size = float(robust_step_size)
        self.generalization_adjustment = generalization_adjustment
        self.automatic_adjustment = bool(automatic_adjustment)
        self.use_normalized_loss = bool(use_normalized_loss)
        self.btl = bool(btl)
        self.minimum_variational_weight = float(minimum_variational_weight)
        self.reweight_groups = bool(reweight_groups)
        self.scheduler = bool(scheduler)
        self.val_fraction = float(val_fraction)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self.n_print = int(n_print)
        self.predict_batch_size = int(kwargs.get("predict_batch_size", 8192))

        if self.model_class is None:
            self.model_class = GenericModel

        self.model = None
        self.optimizer = None
        self.best_state = None
        self.group_values = None
        self.group_to_index = None
        self.input_dim = None
        self.datos_cargados = False
        self.rng = None
        self.torch_generator = None

        self.X_train = None
        self.y_train = None
        self.X_test = None

    def load_data(self, X_train, y_train, X_test):
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.float().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = int(self.X_train.shape[1])
        self.datos_cargados = True

    def _make_group_map(self, sensitive_labels):
        sensitive = np.asarray(sensitive_labels)
        self.group_values = np.unique(sensitive)
        self.group_to_index = {g: i for i, g in enumerate(self.group_values)}
        return np.array([self.group_to_index[g] for g in sensitive], dtype=np.int64)

    def _build_loader(self, X, y, group_idx, train):
        dataset = TensorDataset(
            X,
            y.float(),
            torch.as_tensor(group_idx, dtype=torch.long),
        )
        pin_memory = device == "cuda"

        if train and self.reweight_groups:
            counts = np.bincount(group_idx, minlength=len(self.group_values)).astype(
                np.float64
            )
            counts[counts == 0] = 1.0
            sample_weights = torch.as_tensor(
                (len(group_idx) / counts[group_idx]),
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(
                sample_weights,
                len(sample_weights),
                replacement=True,
                generator=self.torch_generator,
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=pin_memory,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            pin_memory=pin_memory,
            generator=self.torch_generator if train else None,
        )

    def _prepare_fit_and_val_splits(self, g_train_idx):
        train_idx, val_idx = _groupwise_train_val_split(
            g_train_idx, self.val_fraction, self.rng
        )
        X_fit = self.X_train.index_select(
            0, torch.as_tensor(train_idx, dtype=torch.long)
        )
        y_fit = self.y_train.index_select(
            0, torch.as_tensor(train_idx, dtype=torch.long)
        )
        g_fit = g_train_idx[train_idx]

        X_val_t = self.X_train.index_select(
            0, torch.as_tensor(val_idx, dtype=torch.long)
        )
        y_val_t = self.y_train.index_select(
            0, torch.as_tensor(val_idx, dtype=torch.long)
        )
        g_val = g_train_idx[val_idx]
        return X_fit, y_fit, g_fit, X_val_t, y_val_t, g_val

    def _run_epoch(self, loader, loss_computer, training):
        self.model.train(mode=training)
        pin_memory = device == "cuda"

        for xb, yb, gb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            gb = gb.to(device, non_blocking=pin_memory)

            logits = self.model(xb)
            loss = loss_computer.loss(logits, yb, gb)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @staticmethod
    def _worst_observed_group_accuracy(loss_computer):
        observed = loss_computer.processed_data_counts > 0
        if not torch.any(observed):
            return float("-inf")
        return float(torch.min(loss_computer.avg_group_acc[observed]).item())

    def fit(self, sensitive_labels=None, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")
        if sensitive_labels is None:
            raise ValueError("Se requieren etiquetas sensibles para entrenar GroupDRO")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError("val_fraction must be strictly between 0 and 1")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.torch_generator = torch.Generator().manual_seed(self.seed)

        sensitive_np = (
            sensitive_labels.detach().cpu().numpy()
            if isinstance(sensitive_labels, torch.Tensor)
            else np.asarray(sensitive_labels)
        )
        g_train_idx = self._make_group_map(sensitive_np)

        X_fit, y_fit, g_fit, X_val_t, y_val_t, g_val = self._prepare_fit_and_val_splits(
            g_train_idx
        )
        if X_fit.shape[0] == 0 or X_val_t.shape[0] == 0:
            raise ValueError("No hay suficientes datos tras construir train/val para DRO")

        train_loader = self._build_loader(X_fit, y_fit, g_fit, train=True)
        val_loader = self._build_loader(X_val_t, y_val_t, g_val, train=False)

        self.model = self.model_class(self.input_dim).to(device)
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0.0,
                eps=1e-8,
            )
            if self.scheduler
            else None
        )

        train_info = _GroupDatasetInfo(self.group_values, g_fit)
        val_info = _GroupDatasetInfo(self.group_values, g_val)
        adjustments = _parse_generalization_adjustment(
            self.generalization_adjustment, train_info.n_groups
        )

        train_loss_computer = _LossComputer(
            train_info,
            alpha=self.alpha,
            gamma=self.gamma,
            adj=adjustments,
            step_size=self.robust_step_size,
            normalize_loss=self.use_normalized_loss,
            btl=self.btl,
            min_var_weight=self.minimum_variational_weight,
        )

        best_val_acc = float("-inf")
        best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())

        print(
            f"[DRO] Training on {train_info.n_groups} groups "
            f"(epochs={self.n_epochs}, batch_size={self.batch_size}, lr={self.lr}, "
            f"weight_decay={self.weight_decay}, alpha={self.alpha}, "
            f"step_size={self.robust_step_size}, reweight_groups={self.reweight_groups})"
        )

        for epoch in range(self.n_epochs):
            train_loss_computer.reset_stats()
            self._run_epoch(train_loader, train_loss_computer, training=True)

            val_loss_computer = _LossComputer(
                val_info,
                alpha=self.alpha,
                gamma=self.gamma,
                adj=adjustments,
                step_size=self.robust_step_size,
                normalize_loss=self.use_normalized_loss,
                btl=self.btl,
                min_var_weight=self.minimum_variational_weight,
            )
            with torch.no_grad():
                self._run_epoch(val_loader, val_loss_computer, training=False)

            current_worst_val_acc = self._worst_observed_group_accuracy(val_loss_computer)
            if current_worst_val_acc > best_val_acc:
                best_val_acc = current_worst_val_acc
                best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())

            if scheduler is not None:
                if self.btl:
                    val_objective, _ = val_loss_computer.compute_robust_loss_greedy(
                        val_loss_computer.avg_group_loss,
                        val_loss_computer.exp_avg_loss
                        + val_loss_computer.adj / val_loss_computer.group_count_sqrt,
                    )
                else:
                    val_objective, _ = val_loss_computer.compute_robust_loss_greedy(
                        val_loss_computer.avg_group_loss,
                        val_loss_computer.avg_group_loss,
                    )
                scheduler.step(float(val_objective.detach().cpu().item()))

            if self.automatic_adjustment:
                train_loss_computer.adj = (
                    val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
                ) * train_loss_computer.group_count_sqrt
                adjustments = train_loss_computer.adj.detach().cpu().numpy()

            if ((epoch + 1) % self.n_print == 0) or epoch == 0:
                print(
                    f"[DRO] Epoch {epoch + 1}/{self.n_epochs} "
                    f"| train_loss={train_loss_computer.avg_actual_loss.item():.4f} "
                    f"| train_acc={train_loss_computer.avg_acc.item():.4f} "
                    f"| val_worst_acc={current_worst_val_acc:.4f} "
                    f"| val_avg_acc={val_loss_computer.avg_acc.item():.4f}"
                )

        self.model.load_state_dict(self.best_state)
        self.model.eval()
        print(
            f"[DRO] Loaded best checkpoint from epoch {best_epoch} "
            f"(worst-group val accuracy={best_val_acc:.4f})."
        )

    def predict(self, sensitive_labels=None, **kwargs):
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado")

        batch_size = int(kwargs.get("batch_size", self.predict_batch_size))
        loader = DataLoader(
            TensorDataset(self.X_test),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

        self.model.eval()
        chunks = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=(device == "cuda"))
                logits = self.model(xb)
                chunks.append(_positive_class_probs(logits).cpu())
        probs = torch.cat(chunks, dim=0)
        return probs.unsqueeze(1).numpy()


DRO = GroupDRO
