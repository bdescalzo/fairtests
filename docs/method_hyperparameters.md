# Fair Method Hyperparameters

This document summarizes the hyperparameters accepted by the active fairness methods exposed through `run_fairtests(...)`.

Method keys registered in `fairtests.py` (`AVAILABLE_METHODS`):

- `baseline` -> `fair_methods.baseline.Baseline`
- `dro` -> `fair_methods.dro.GroupDRO`
- `maml` -> `fair_methods.meta.MetaLearning`
- `reptile` -> `fair_methods.reptile.Reptile`
- `mmpf` -> `fair_methods.mmpf.MinimaxParetoFairness`

## Common Notes

- `DRO = GroupDRO` and `MinimaxParetoFair = MinimaxParetoFairness` exist as module/package aliases, but those alias names are not valid `method_names` in `run_fairtests(...)` unless you pass them yourself through `methods=`.
- All active methods read `predict_batch_size` from extra constructor `**kwargs`. This affects `predict(...)`, not training.
- `run_fairtests(...)` always instantiates methods with `model_class=...`. If you omit `model_class`, it injects `models.GenericModel`.
- Direct instantiation is different from `run_fairtests(...)`: `Baseline`, `GroupDRO`, `MetaLearning`, and `Reptile` do not implement an internal fallback network when `model_class=None`; `MinimaxParetoFairness` does.
- `fit(...)` always receives `sensitive_train` from `run_fairtests(...)`.

## `Baseline`

Constructor:

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `lr` | `1e-3` | Adam learning rate. |
| `epochs` | `15` | Number of training epochs. |
| `batch_size` | `1024` | Mini-batch size for supervised training. |
| `seed` | `42` | Random seed used before model initialization and batch shuffling. |
| `model_class` | `None` | Compatible model class. `run_fairtests(...)` normally injects `GenericModel`; direct construction requires a non-`None` class by `fit(...)` time. |
| `predict_batch_size` | `8192` | Optional extra kwarg consumed during prediction batching. |

Other accepted call-time arguments:

- `fit(sensitive_labels=None, **kwargs)`: ignores `sensitive_labels`.
- `predict(batch_size=...)`: optional override for prediction batch size.

## `GroupDRO`

Constructor:

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `n_epochs` | `100` | Number of training epochs. |
| `batch_size` | `128` | Mini-batch size. |
| `lr` | `1e-3` | Optimizer learning rate. |
| `weight_decay` | `1e-4` | SGD weight decay. |
| `alpha` | `0.2` | Robust-loss mass used by the loss computer. |
| `gamma` | `0.1` | Exponential moving-average update rate for group losses. |
| `robust_step_size` | `0.01` | Step size for adversarial group reweighting. |
| `generalization_adjustment` | `0.0` | Scalar or per-group adjustment term. |
| `automatic_adjustment` | `False` | Enables automatic adjustment from validation losses. |
| `use_normalized_loss` | `False` | Normalizes robust loss inputs before the adversarial update. |
| `btl` | `False` | Enables the greedy/bottom-k robust objective path. |
| `minimum_variational_weight` | `0.0` | Minimum weight retained in the BTL robust weighting rule. |
| `reweight_groups` | `True` | Uses a weighted sampler to rebalance groups during training. |
| `scheduler` | `False` | Enables `ReduceLROnPlateau` on the validation robust-loss objective. |
| `val_fraction` | `0.2` | Fraction used for internal groupwise train/validation splitting. |
| `momentum` | `0.9` | SGD momentum. |
| `seed` | `42` | Random seed. |
| `n_print` | `10` | Logging cadence in epochs. |
| `model_class` | `None` | Compatible model class. `run_fairtests(...)` normally injects `GenericModel`; direct construction requires a non-`None` class by `fit(...)` time. |
| `predict_batch_size` | `8192` | Optional extra kwarg consumed during prediction batching. |

Other accepted call-time arguments:

- `fit(sensitive_labels, **kwargs)`: requires sensitive labels.
- `predict(batch_size=...)`: optional override for prediction batch size.

## `MetaLearning` (`maml`)

Constructor:

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `inner_lr` | `0.005` | Inner-loop adaptation step size. |
| `inner_steps` | `5` | Number of gradient steps in each task adaptation. |
| `meta_epochs` | `150` | Number of outer-loop meta-training epochs. |
| `meta_lr` | `3e-4` | Outer-loop Adam learning rate. |
| `k_support` | `128` | Requested support set size per sensitive-group task. This is used as-is only when `replace=True`. |
| `k_query` | `128` | Requested query set size per sensitive-group task. This is used as-is only when `replace=True`. |
| `replace` | `False` | Allows sampling with replacement when task episodes are built. |
| `group_budget_divisor` | `4` | When `replace=False`, sets the shared per-group task budget to `max(2, min_group_size // group_budget_divisor)`. |
| `support_fraction` | `1/3` | When `replace=False`, splits the effective task budget between support and query examples. |
| `use_full_data` | `False` | During final per-group adaptation, uses all group samples instead of only support samples. |
| `seed` | `42` | Random seed. |
| `model_class` | `None` | Compatible model class. `run_fairtests(...)` normally injects `GenericModel`; direct construction requires a non-`None` class by `fit(...)` time. |
| `predict_batch_size` | `8192` | Optional extra kwarg consumed during prediction batching. |

Other accepted call-time arguments:

- `fit(sensitive_labels, **kwargs)`: requires sensitive labels.
- `predict(sensitive_labels=..., batch_size=...)`: test-time sensitive labels are optional but important if you want per-group adapted parameters to be used.

Important behavior:

- If `replace=False`, the effective `k_support` and `k_query` are recomputed from the smallest sensitive-group size, `group_budget_divisor`, and `support_fraction`; they are not simple clipped versions of the requested constructor values.

## `Reptile`

Constructor:

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `inner_lr` | `0.005` | Inner-loop Adam learning rate. |
| `inner_steps` | `10` | Number of inner-loop update steps per episode. |
| `meta_epochs` | `200` | Number of outer-loop meta-training epochs. |
| `meta_lr` | `0.003` | Initial meta step size. |
| `meta_lr_final` | `None` | Final meta step size; if omitted, stays equal to `meta_lr`. |
| `inner_batch_size` | `128` | Inner-loop batch size during meta-training. |
| `k_support` | `128` | Episode size used for final per-group adaptation. |
| `train_k_support` | `None` | Episode size used during meta-training; defaults to `k_support`. |
| `meta_batch_size` | `4` | Number of tasks sampled per outer-loop iteration. |
| `replace` | `False` | Enables replacement-based episode sampling. |
| `eval_inner_steps` | `None` | Inner steps used for final per-group adaptation; defaults to `inner_steps`. |
| `eval_inner_batch_size` | `None` | Batch size used for final per-group adaptation; defaults to `inner_batch_size`. |
| `seed` | `42` | Random seed. |
| `model_class` | `None` | Compatible model class. `run_fairtests(...)` normally injects `GenericModel`; direct construction requires a non-`None` class by `fit(...)` time. |
| `predict_batch_size` | `8192` | Optional extra kwarg consumed during prediction batching. |

Other accepted call-time arguments:

- `fit(sensitive_labels, **kwargs)`: requires sensitive labels.
- `predict(sensitive_labels=..., batch_size=...)`: test-time sensitive labels are optional but needed to route examples through group-adapted parameters.

Important behavior:

- If `replace=False`, only `train_k_support` and `k_support` are clipped down to the smallest observed sensitive-group size.
- If `replace=True`, `inner_batch_size` and `eval_inner_batch_size` are capped so they do not exceed the corresponding episode sizes.

## `MinimaxParetoFairness` (`mmpf`)

Constructor:

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `lr` | `5e-4` | Adam learning rate. |
| `max_epochs` | `1000` | Maximum epochs for each inner adaptive-optimizer phase. |
| `batch_size` | `32` | Mini-batch size. |
| `niter` | `5` | Number of outer APSTAR-like penalty updates. |
| `patience` | `10` | Early-stopping patience for both adaptive training and outer iterations. |
| `lrdecay` | `0.25` | Learning-rate decay factor. |
| `alpha` | `0.5` | Update weight applied to the fairness penalty vector `mu`. |
| `k_ini` | `1` | Initial APSTAR denominator term. |
| `k_min` | `20` | Upper bound used when resetting/improving `K`. |
| `risk_round_factor` | `3` | Decimal precision used before risk comparisons. |
| `reset_optimizer` | `False` | Recreates the optimizer at each outer iteration. |
| `hidden_units` | `(512, 512)` | Hidden layer widths for the internal `_MMPFNet`; only used when `model_class is None`. |
| `balanced_sampler` | `True` | Uses group-balanced weighted sampling for the training loader. |
| `seed` | `42` | Random seed. |
| `n_print` | `5` | Logging cadence during the adaptive optimizer phase. |
| `val_fraction` | `0.2` | Fraction used for internal groupwise train/validation splitting. |
| `model_class` | `None` | If provided, builds `model_class(input_dim, output_dim=2)`; otherwise falls back to the internal `_MMPFNet`. |
| `predict_batch_size` | `8192` | Optional extra kwarg consumed during prediction batching. |

Other accepted call-time arguments:

- `fit(sensitive_labels, **kwargs)`: requires sensitive labels.
- `predict(batch_size=...)`: optional override for prediction batch size.

Important behavior:

- Through `run_fairtests(...)`, this method normally receives `model_class=GenericModel`, so the internal `_MMPFNet` fallback and `hidden_units` only matter when you instantiate `MinimaxParetoFairness` yourself with `model_class=None`.
