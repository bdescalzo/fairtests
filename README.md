# fairtests: Mini-Max Fairness Testing framework

This project is a minimal framework for testing different minimax-focused fair learning algorithms, and comparing them against meta-learning approaches.

Fairtests is built as part of an internship in the Intelligent Systems Group of the UPV/EHU, and aims to test whether MAML meta-learning (and derivatives like Reptile) can compete in mini-max fairness against other fairness-aware methods.

## Usage

The ```examples\``` folder contains a few usage examples. The package is straightforward to use:
1. For each split (train/test), divide your data in three Pytorch tensors: X, y and g, where g[i] contains the label of the i-th sample.
2. Call fairtests' ```run_fairtests``` method with the tensors, the protected label and the list of methods to execute. It will return a dictionary with all the metrics per method (as can be seen in the ```print_toy_example.py``` example).

The ```results_excel.py``` script under examples generates a XLSX file with all the data conveniently formatted in sheets.

## Examples

The ```examples\``` folder contains ```example.py``` (which uses the [folktables](https://github.com/socialfoundations/folktables) dataset), and ```example_toy.py```, which generates a simple two-variable Gaussian distribution.

## Method attributions

The implemented learning methods come from the following papers:

1. ```meta.py```: Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In Proceedings of the 34th International Conference on Machine Learning
    (PMLR 70). arXiv:1703.03400v3.
  - ```reptile.py```: Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. arXiv:1803.02999v3.
  - ```mmpf.py```: Martinez, N., Bertran, M., & Sapiro, G. (2020). Minimax Pareto Fairness: A Multi Objective Perspective. Proceedings of Machine Learning Research, 119, 6755-6764.
