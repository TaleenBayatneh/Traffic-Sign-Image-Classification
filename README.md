# Traffic Sign Image Classification

A comparative study of three machine learning classifiers applied to traffic sign recognition.

## Overview

This project trains and evaluates three classifiers on a dataset of traffic sign images across three categories:

| Class | Description |
|-------|-------------|
| `stop` | Stop signs |
| `no_entry` | No-entry signs |
| `speed_limit` | Speed limit signs |

The three models compared are:

- **Naive Bayes** - probabilistic baseline
- **Decision Tree** - interpretable tree-based classifier
- **MLP (Feedforward Neural Network)** - multi-layer perceptron

## Project Structure

- `main.py` – entry point; loads data, runs selected models, and triggers cross-validation
- `helpers/data_loader.py` – image loading and preprocessing (resize, flatten, label encoding)
- `helpers/eval_tools.py` – saves classification reports (CSV) and confusion matrix plots (PNG)
- `helpers/cv_grid.py` – 5-fold cross-validation utilities
- `models/naive_bayes.py` – Naive Bayes classifier implementation
- `models/decision_tree.py` – Decision Tree classifier implementation
- `models/mlp_model.py` – MLP (Feedforward Neural Network) implementation
- `all_data/` – raw dataset; class subfolders: `stop/`, `no_entry/`, `speed_limit/`
- `train/` – auto-generated 80% training split (created on first run)
- `test/` – auto-generated 20% testing split (created on first run)
- `outputs/` – generated CSV classification reports and PNG confusion matrix plots
- `requirements.txt` – Python dependencies

## Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`


### CLI Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--model` | `all`, `nb`, `dt`, `mlp` | `all` | Which model(s) to train and evaluate |
| `--size` | any integer | `32` | Resize images to SIZE x SIZE pixels |
| `--cv` | flag | off | Run 5-fold cross-validation after training |

## Outputs

All results are saved to the `outputs/` folder:

| File | Description |
|------|-------------|
| `<model>_classification_report.csv` | Per-class precision, recall, F1-score |
| `<model>_confusion_matrix.png` | Confusion matrix heatmap |
| `decision_tree_structure.png` | Visual diagram of the trained decision tree |
