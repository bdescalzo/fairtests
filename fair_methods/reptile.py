import copy
import random

import numpy as np
import torch
from torch import nn

from .fair_method import FairMethod

device = "cuda" if torch.cuda.is_available() else "cpu"


class Reptile(FairMethod):
    def __init__(
        self,
        inner_lr=0.005,
        inner_steps=10,
        meta_epochs=200,
        meta_lr=0.003,
        meta_lr_final=None,
        inner_batch_size=128,
        k_support=128,
        train_k_support=None,
        meta_batch_size=4,
        replace=False,
        eval_inner_steps=None,
        eval_inner_batch_size=None,
        seed=42,
        model_class=None,
        **kwargs,
    ):
        super().__init__(model_class=model_class, **kwargs)
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.meta_epochs = int(meta_epochs)
        self.meta_lr = float(meta_lr)
        self.meta_lr_final = float(meta_lr if meta_lr_final is None else meta_lr_final)
        self.inner_batch_size = int(inner_batch_size)
        self.k_support = int(k_support)
        self.train_k_support = int(
            self.k_support if train_k_support in (None, 0) else train_k_support
        )
        self.meta_batch_size = int(meta_batch_size)
        self.replace = bool(replace)
        self.eval_inner_steps = (
            self.inner_steps if eval_inner_steps is None else int(eval_inner_steps)
        )
        self.eval_inner_batch_size = (
            self.inner_batch_size
            if eval_inner_batch_size is None
            else int(eval_inner_batch_size)
        )
        self.seed = seed

        self.meta_model = None
        self.group_params = {}
        self.group_indices = {}
        self.datos_cargados = False
        self.input_dim = None
        self.sensitive_train = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.predict_batch_size = kwargs.get("predict_batch_size", 8192)
        self.rng = None

        self.effective_train_k_support = self.train_k_support
        self.effective_k_support = self.k_support
        self.effective_inner_batch_size = self.inner_batch_size
        self.effective_eval_inner_batch_size = self.eval_inner_batch_size
        self._working_model = None
        self._working_optimizer = None
        self._batch_rng = None

    def load_data(self, X_train, y_train, X_test):
        # Keep full datasets on CPU and stream episode batches to the device.
        self.X_train = X_train.float().cpu()
        self.y_train = y_train.float().cpu()
        self.X_test = X_test.float().cpu()
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    def _initialize_model(self, model):
        # Match the original Reptile comparison wrapper's Glorot-uniform
        # linear initialization with zero biases.
        linear_idx = 0
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, nn.Linear):
                    continue
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed + linear_idx)
                fan_out, fan_in = module.weight.shape
                bound = float(np.sqrt(6.0 / float(fan_in + fan_out)))
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.zero_()
                linear_idx += 1

    def _resolve_effective_episode_settings(self, unique_groups):
        positive_ints = {
            "inner_steps": self.inner_steps,
            "meta_epochs": self.meta_epochs,
            "inner_batch_size": self.inner_batch_size,
            "k_support": self.k_support,
            "train_k_support": self.train_k_support,
            "meta_batch_size": self.meta_batch_size,
            "eval_inner_steps": self.eval_inner_steps,
            "eval_inner_batch_size": self.eval_inner_batch_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        counts = [self.group_indices[g_id].size for g_id in unique_groups]
        min_group_size = int(np.min(counts))
        if min_group_size < 1:
            raise ValueError("All sensitive groups must contain at least one sample.")

        self.effective_train_k_support = int(self.train_k_support)
        self.effective_k_support = int(self.k_support)
        if not self.replace:
            # Keep a shared episode size across groups when replacement is disabled.
            self.effective_train_k_support = min(
                self.effective_train_k_support, min_group_size
            )
            self.effective_k_support = min(self.effective_k_support, min_group_size)

        self.effective_inner_batch_size = int(self.inner_batch_size)
        self.effective_eval_inner_batch_size = int(self.eval_inner_batch_size)
        if self.replace:
            # The reference implementation samples full batches independently from
            # the episode, so replacement mode cannot exceed the episode size.
            self.effective_inner_batch_size = min(
                self.effective_inner_batch_size, self.effective_train_k_support
            )
            self.effective_eval_inner_batch_size = min(
                self.effective_eval_inner_batch_size, self.effective_k_support
            )

    def _sample_episode_indices(self, group_id, episode_size):
        idxs = self.group_indices.get(group_id)
        if idxs is None or idxs.size == 0:
            return None

        replace = self.replace and idxs.size < episode_size
        chosen = self.rng.choice(idxs, size=episode_size, replace=replace)
        return np.asarray(chosen, dtype=np.int64)

    def _iter_episode_batches(self, episode_indices, batch_size, num_batches):
        samples = [int(sample) for sample in np.asarray(episode_indices, dtype=np.int64)]
        if not samples or num_batches <= 0:
            return

        # Match supervised Reptile's batching RNG structure: use a dedicated
        # Python RNG stream for inner-loop mini-batches while NumPy drives task
        # and episode sampling.
        if self.replace:
            for _ in range(num_batches):
                yield np.asarray(
                    self._batch_rng.sample(samples, batch_size), dtype=np.int64
                )
            return

        cur_batch = []
        batch_count = 0
        while True:
            shuffled = list(samples)
            self._batch_rng.shuffle(shuffled)
            for sample in shuffled:
                cur_batch.append(sample)
                if len(cur_batch) < batch_size:
                    continue
                yield np.asarray(cur_batch, dtype=np.int64)
                cur_batch = []
                batch_count += 1
                if batch_count == num_batches:
                    return

    def _adapt_on_episode(self, model, episode_indices, inner_steps, inner_batch_size):
        if episode_indices is None or episode_indices.size == 0:
            return

        model.train()
        for batch_indices in self._iter_episode_batches(
            episode_indices, inner_batch_size, inner_steps
        ):
            idx_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
            Xb = self.X_train.index_select(0, idx_tensor).to(device)
            yb = self.y_train.index_select(0, idx_tensor).to(device)
            self._working_optimizer.zero_grad()
            logits = model(Xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self._working_optimizer.step()

    def _clone_model_state(self, model):
        return {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }

    def _load_working_state(self, model_state, optimizer_state=None):
        self._working_model.load_state_dict(model_state)
        if optimizer_state is not None:
            self._working_optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    def _current_meta_lr(self, epoch):
        frac_done = epoch / float(self.meta_epochs)
        return frac_done * self.meta_lr_final + (1.0 - frac_done) * self.meta_lr

    def fit(self, sensitive_labels, **kwargs):
        if not self.datos_cargados:
            raise RuntimeError("No hay datos de entrenamiento cargados")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self._batch_rng = random.Random(self.seed)

        # Update loss to handle class imbalance in the training data.
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

        self.group_indices = {
            g_id: np.flatnonzero(self.sensitive_train == g_id) for g_id in unique_groups
        }
        self._resolve_effective_episode_settings(unique_groups)

        self.meta_model = self.model_class(self.input_dim)
        self._initialize_model(self.meta_model)
        self.meta_model = self.meta_model.to(device)
        self._working_model = self.model_class(self.input_dim).to(device)
        self._working_model.load_state_dict(self.meta_model.state_dict())
        self._working_optimizer = torch.optim.Adam(
            self._working_model.parameters(), lr=self.inner_lr, betas=(0.0, 0.999)
        )

        print(
            f"[Reptile] Meta-training on {unique_groups.size} groups "
            f"(epochs={self.meta_epochs}, inner_steps={self.inner_steps}, "
            f"inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"meta_lr_final={self.meta_lr_final}, "
            f"train_k_support={self.effective_train_k_support}, "
            f"inner_batch_size={self.effective_inner_batch_size}, "
            f"replace={self.replace})"
        )

        for epoch in range(self.meta_epochs):
            old_model_state = self._clone_model_state(self.meta_model)
            task_ids = self.rng.choice(
                unique_groups, size=self.meta_batch_size, replace=True
            )
            deltas = [torch.zeros_like(param) for param in self.meta_model.parameters()]
            valid_tasks = 0

            for group_id in task_ids:
                self._load_working_state(old_model_state)
                episode_indices = self._sample_episode_indices(
                    group_id, self.effective_train_k_support
                )
                if episode_indices is None:
                    continue

                self._adapt_on_episode(
                    self._working_model,
                    episode_indices,
                    self.inner_steps,
                    self.effective_inner_batch_size,
                )

                with torch.no_grad():
                    for i, (meta_param, adapted_param) in enumerate(
                        zip(self.meta_model.parameters(), self._working_model.parameters())
                    ):
                        deltas[i] += adapted_param.data - meta_param.data
                valid_tasks += 1

            if valid_tasks == 0:
                continue

            current_meta_lr = self._current_meta_lr(epoch)
            with torch.no_grad():
                scale = current_meta_lr / float(valid_tasks)
                for meta_param, delta in zip(self.meta_model.parameters(), deltas):
                    meta_param.data += scale * delta
            self._load_working_state(self._clone_model_state(self.meta_model))

            if (epoch + 1) % 20 == 0:
                print(
                    f"[Reptile] Meta-epoch {epoch + 1}/{self.meta_epochs} "
                    f"| meta_lr={current_meta_lr:.6f}"
                )

        print("[Reptile] Adapting per-group parameters...")
        base_model_state = self._clone_model_state(self.meta_model)
        base_optimizer_state = copy.deepcopy(self._working_optimizer.state_dict())
        self.group_params = {}
        for g_id in unique_groups:
            self._load_working_state(base_model_state, base_optimizer_state)
            episode_indices = self._sample_episode_indices(
                g_id, self.effective_k_support
            )
            self._adapt_on_episode(
                self._working_model,
                episode_indices,
                self.eval_inner_steps,
                self.effective_eval_inner_batch_size,
            )
            self.group_params[g_id] = {
                name: param.detach().cpu()
                for name, param in self._working_model.named_parameters()
            }
        self._load_working_state(base_model_state, base_optimizer_state)

        self.meta_model.eval()
        print(
            "[Reptile] Adaptation complete "
            f"(k_support={self.effective_k_support}, "
            f"eval_inner_steps={self.eval_inner_steps}, "
            f"eval_inner_batch_size={self.effective_eval_inner_batch_size})."
        )

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
                    params = {name: param.to(device) for name, param in params_cpu.items()}

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
