# Kernel Methods Data Challenge Submission

## Overview

This archive contains two versions of our code:

- `start.py`  
  Required entry-point script for the submission. Running this file generates the final prediction file `Yte.csv`.

- `t2.py`  
  Final submission version used by `start.py`. This script directly runs our final ensemble using three hardcoded configurations selected from previous search runs. It also performs a quick validation check and plots a confusion matrix, then trains on the full training set and writes `Yte.csv` in the same directory.

- `t1.py`  
  Full experimental version of the project. This script runs the complete feature-building and model-selection pipeline, including the search over multiple candidate configurations, validation comparison, and optional final ensembling. It is included for completeness and reproducibility of our development procedure, but it is much slower than the final submission version.

## Required data files

Please add the following CSV files to the same folder:

- `Xtr.csv`  
- `Xte.csv`  
- `Ytr.csv`

## Folder structure

The directory should look like:

- `start.py`  
- `t2.py`  
- `t1.py`  
- `Xtr.csv`  
- `Xte.csv`  
- `Ytr.csv`

## How to run

From this directory, run:

```bash
python start.py
