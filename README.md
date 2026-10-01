# CSCI 114: Stroke risk feature representations

This repository contains our CSCI 114 Lab 2 work on dimensionality reduction. We use a stroke risk dataset to compare PCA with an autoencoder and study how much information each method keeps at different representation sizes.

Start with [the Lab 2 notebook](lab2/lab2.ipynb). It contains the dataset notes, preprocessing code, and candidate dimensions for the experiments.

## Current progress

The notebook currently covers data loading, a stratified train/validation/test split, missing-value imputation, numerical scaling, and one-hot encoding. The processed data has 89 input features, and the chosen reduced dimensions are `2`, `15`, `45`, and `70`.

The PCA and autoencoder experiments are still pending. The notebook does not yet contain a comparison of their results.

## Files

| File | What it contains |
| --- | --- |
| [lab2/lab2.ipynb](lab2/lab2.ipynb) | Main notebook for the lab |
| [lab2/stroke_risk_prediction_dataset.csv](lab2/stroke_risk_prediction_dataset.csv) | Dataset used by the notebook |
| [lab2/lab2.pdf](lab2/lab2.pdf) | Lab PDF |

## Run the notebook

You need Python 3 and Git. From a terminal:

```bash
git clone https://github.com/Gabriel-Labariento/csci114.git
cd csci114
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the notebook dependencies and open JupyterLab:

```bash
python -m pip install jupyterlab numpy pandas matplotlib scikit-learn torch
cd lab2
python -m jupyterlab lab2.ipynb
```

Run the cells from top to bottom. Keep the notebook and CSV in the same directory because the notebook reads the dataset using a relative path. You can also open the notebook in VS Code and select the `.venv` Python environment as its kernel.

## Dataset and preprocessing

The CSV has 50,000 records and 40 columns. The notebook identifies its source as the [Stroke Risk Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/mobeenfatimah/stroke-risk-prediction-dataset).

The target is `Stroke_Risk`, with the classes `Low`, `Moderate`, and `High`. Most records belong to `Moderate`, so the split uses stratification to keep similar class proportions across the three partitions.

| Partition | Share | Records |
| --- | --- | --- |
| Training | 70% | 35,000 |
| Validation | 15% | 7,500 |
| Test | 15% | 7,500 |

The notebook excludes `Patient_ID`, `Stroke_Risk_Score`, `AI_Health_Recommendation`, and `Doctor_Consultation_Needed` from the inputs. These are an identifier or columns that may reveal the target. After removing them and separating the target, 35 input columns remain: 16 numerical and 19 categorical.

Numerical columns use training-set medians for missing values and `StandardScaler` for scaling. Missing values in `Medication_Adherence` become `"Missing"`. `OneHotEncoder` converts the categorical columns to binary columns and ignores categories it did not see during fitting.

Both transformers fit only on the training data. Validation and test data use the same fitted transformers. The resulting arrays are `X_train_processed`, `X_val_processed`, and `X_test_processed`; `feature_names` lists their columns in order.

## Representation sizes

Here, `n` is the number of input features after preprocessing, and `k` is the size of the reduced representation. PCA uses `k` components; the autoencoder uses `k` neurons in its bottleneck layer.

Use the same candidates for both methods:

```python
n = X_train_processed.shape[1]  # 89 with the current preprocessing
candidate_dimensions = [2, 15, 45, 70]
```

Each candidate satisfies `1 <= k < n` and fits within the available training sample count. The range lets us compare strong compression with representations that retain more dimensions.

## Working together

Pull the latest changes before editing. When checking a notebook change, restart the kernel and run all cells so the results do not depend on variables left over from an earlier run. Keep the preprocessing fitted on training data only when adding the remaining experiments.

README author: Gabriel Matthew Labariento.
