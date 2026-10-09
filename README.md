[README.md](https://github.com/user-attachments/files/33268393/README.md)
# Simulation Latency Regression

Predict simulated latency from user load, payload size and CPU utilization.

## Tools

Python, NumPy, pandas, scikit-learn

## Workflow

1. Generate 800 synthetic observations with numerical features and missing values.
2. Split data into training and test sets.
3. Compare a mean baseline, linear regression and random forest regression.
4. Fit median imputation within each pipeline.
5. Export test predictions and MAE, RMSE and R² metrics.

## Data and method

Uses synthetic data generated from a fixed mathematical relationship. This Python project runs independently of Unity.

## Run locally

```bash
python -m pip install -r requirements.txt
python main.py
```

For the package versions used to verify the project, install `requirements-tested.txt` instead.

## Run in Google Colab

Upload `project.ipynb` to Google Colab and select **Run all**.

## Files

- `main.py`: executable Python script.
- `project.ipynb`: notebook with the same workflow and example outputs.
- `requirements.txt`: required Python packages.
- `requirements-tested.txt`: package versions used for verification.
- `outputs/`: generated data, reports and results.

## About

This is a learning project prepared with AI assistance.
