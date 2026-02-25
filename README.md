# fairtests: Mini-Max Fairness Testing framework

This project is a minimal framework for testing different SOTA fair learning algorithms, and comparing them against meta-learning approaches.

Fairtests is built as part of an internship in the Intelligent Systems Group of the UPV/EHU, and aims to test whether MAML meta-learning (and derivatives) can compete in mini-max fairness against other fairness-aware methods.

## Usage

The ```examples\``` folder contains a few usage examples. The package is straightforward to use:
1. For each split (train/test), divide your data in three Pytorch tensors: X, y and g, where g[i] contains the label of the i-th sample.
2. Call fairtests' ```run_fairtests``` method with the tensors, the protected label and the list of methods to execute. It will return a dictionary with all the metrics per method (as can be seen in the ```print_toy_example.py``` example).

The ```results_excel.py``` script under examples generates a XLSX file with all the data conveniently formatted in sheets.