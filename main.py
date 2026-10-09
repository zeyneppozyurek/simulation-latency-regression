from pathlib import Path
import json
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
rng = np.random.default_rng(42)
n = 800
X = pd.DataFrame({"concurrent_users": rng.integers(1, 100, n), "payload_kb": rng.uniform(5, 300, n), "cpu_usage_pct": rng.uniform(10, 95, n)})
y = 10 + 0.6 * X.concurrent_users + 0.08 * X.payload_kb + 0.3 * X.cpu_usage_pct + 0.006 * X.concurrent_users ** 2 + rng.normal(0, 6, n)
X.loc[rng.choice(n, 20, replace=False), "payload_kb"] = np.nan
data = X.assign(latency_ms=y)
data.to_csv(OUT / "synthetic_load_data.csv", index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"mean_baseline": DummyRegressor(), "linear_regression": LinearRegression(), "random_forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42)}
metrics = {}
predictions = pd.DataFrame({"actual_latency_ms": y_test.to_numpy()})
for name, estimator in models.items():
    pipeline = make_pipeline(SimpleImputer(strategy="median"), estimator)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    predictions[name] = pred
    metrics[name] = {"mae_ms": mean_absolute_error(y_test, pred), "rmse_ms": float(np.sqrt(mean_squared_error(y_test, pred))), "r2": r2_score(y_test, pred)}
assert len(X_train) == 640 and len(X_test) == 160
predictions.to_csv(OUT / "test_predictions.csv", index=False)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
