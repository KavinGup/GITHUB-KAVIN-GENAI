from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

# 1. Tracking URI aur Experiment set karein (port 6001 aur sqlite backend ke mutabiq)
mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("Iris-MLOps-Experiment")

# 2. Dataset load karein
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

n_estimators = 1
max_depth = 2
random_state = 0
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "random_state": random_state,
    "test_size": 0.25
}

# 3. Start a tracked training run using MLflow
with mlflow.start_run() as run:
    # Parameters log karein
    mlflow.log_params(model_parameters)

    # Model initialize aur train karein
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)  # Fix: X_trainy_train -> X_train, y_train

    # Predictions aur metrics calculate karein
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # Metrics log karein
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # MLflow ke andar model log/artifact save karein
    mlflow.sklearn.log_model(
        model,
        "iris-random-forest-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Accuracy logged to MLflow: {accuracy}")

# 4. Local Joblib file save (jaise aapke purane pipeline mein thi)
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_dir / "iris_model.joblib")
print(f"Model saved locally to {model_dir / 'iris_model.joblib'}")