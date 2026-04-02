# fairtests: Minimax Group Fairness testing framework

This project is a minimal framework for testing different minimax-fairness focused learning algorithms. It aims to compare different MAML-based approaches to state-of-the-art minimax-fairness algorithms in various binary classification tasks.

Fairtests was developed as part of a research internship at the Intelligent Systems Group (ISG) in EHU.

## Usage

The ```examples/``` folder contains a few usage examples. The package is straightforward to use:
1. For each split (train/test), divide your data in three PyTorch tensors: X, y and g, where g[i] contains the label of the i-th sample for the protected attribute. You should remove the protected attribute from the X tensor.
    * You can additionally build and pass X_train_full and X_test_full, which will be assumed to contain the protected attribute as an explicit feature. If these are passed as parameters to run_fairtests, some methods will run twice (right now only the baseline ERM), offering the comparison between access and lack thereof to the protected attribute.

2. Call fairtests' ```run_fairtests``` method with the tensors and the list of methods to execute. It will return a dictionary with all the metrics per method.
    * Note that all standard fairness metrics are for the worst pairwise values obtained.

The ```results_excel.py``` script under examples generates a XLSX file with all the data conveniently formatted in sheets.

## Examples

The ```examples/``` folder contains ```example.py```, which uses the [folktables](https://github.com/socialfoundations/folktables) dataset (and ```nine_states.py``` for the same task limited to a subset of states), and ```example_toy.py```, which generates a simple two-variable Gaussian distribution.

## TODO / Next planned steps
1. Add more SOTA minimax-fairness learning algorithms found in the literature.
2. Allow for easier hyperparameter choice, and implement automatic tuning for each method.
3. Allow for multiple runs with different seeds for cross-validation.
4. Implement clustering-based methods for MAML.
5. Implement standardized tests for various specific scenarios.

## Method attributions

The implemented learning methods come from the following papers:

1. ```meta.py```: Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In Proceedings of the 34th International Conference on Machine Learning
    (PMLR 70). arXiv:1703.03400v3.
2. ```reptile.py```: Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. arXiv:1803.02999v3.
3. ```mmpf.py```: Martinez, N., Bertran, M., & Sapiro, G. (2020). Minimax Pareto Fairness: A Multi Objective Perspective. Proceedings of Machine Learning Research, 119, 6755-6764.
4. ```dro.py```: Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. In International Conference on Learning Representations (ICLR).
