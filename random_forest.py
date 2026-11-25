import os
import pickle
import numpy as np
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from astar import astar_recommend

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "random_forest_model.pkl")
DATASET_PATH = os.path.join(PROJECT_DIR, "data", "Crop_recommendation.csv")
LEGACY_DATASET_PATH = os.path.join(PROJECT_DIR, "crop_recommendation.csv")


def resolve_dataset_path() -> str:
    if os.path.exists(DATASET_PATH):
        return DATASET_PATH
    if os.path.exists(LEGACY_DATASET_PATH):
        return LEGACY_DATASET_PATH
    raise FileNotFoundError(
        "Crop recommendation dataset was not found. Place Crop_recommendation.csv "
        "in the data/ folder or crop_recommendation.csv in the project folder."
    )


def _dataset_fingerprint(dataset_path: str) -> str:
    digest = hashlib.sha256()
    with open(dataset_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train_and_save(dataset_path: str = None) -> dict:
    """Train the RF model, save to disk, return metrics."""
    dataset_path = resolve_dataset_path() if dataset_path is None else dataset_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if not {"label", *FEATURE_COLS}.issubset(df.columns):
        raise ValueError("Dataset must contain N, P, K, temperature, humidity, ph, rainfall, and label columns")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if df["label"].nunique() != 22:
        raise ValueError("Dataset must contain exactly 22 crop classes")
    if len(df) != 2200:
        raise ValueError(f"Dataset must contain 2200 records; found {len(df)}")

    required = set(FEATURE_COLS)
    if df[list(required)].select_dtypes(include=["number"]).shape != df[list(required)].shape:
        raise ValueError("All feature columns must contain numeric values")

    dataset_path = os.path.abspath(dataset_path)
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    bundle = {
        "model": clf,
        "label_encoder": le,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": list(le.classes_),
        "dataset_fingerprint": _dataset_fingerprint(dataset_path),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "accuracy": round(acc * 100, 2),
        "report":   classification_report(y_test, y_pred, target_names=le.classes_),
        "classes":  list(le.classes_),
    }


def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            dataset_path = resolve_dataset_path()
            if bundle.get("dataset_fingerprint") == _dataset_fingerprint(dataset_path):
                return bundle
    except (OSError, pickle.PickleError, EOFError, AttributeError):
        pass
    train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def rf_predict(input_features: dict, top_k: int = 5) -> list:
    """
    Predict crop probabilities using Random Forest.

    Parameters
    ----------
    input_features : dict with keys matching FEATURE_COLS
    top_k          : number of top predictions to return

    Returns
    -------
    List of dicts: [{crop, probability, rank}, ...]
    """
    bundle = load_model()
    clf = bundle["model"]
    le  = bundle["label_encoder"]

    X = np.array([[input_features[f] for f in FEATURE_COLS]])
    proba = clf.predict_proba(X)[0]

    top_indices = np.argsort(proba)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "crop":        le.classes_[idx],
            "probability": round(float(proba[idx]) * 100, 2),
            "rank":        rank,
        })
    return results


def get_accuracy() -> float:
    bundle = load_model()
    return bundle.get("accuracy", 0.0)


def get_evaluation() -> dict:
    bundle = load_model()
    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLS]
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    astar_predictions = []
    astar_nodes = []
    for _, row in X_test.iterrows():
        input_features = {feature: row[feature] for feature in FEATURE_COLS}
        result = astar_recommend(input_features, top_k=1)[0]
        astar_predictions.append(result["crop"])
        astar_nodes.append(result["nodes_expanded"])

    astar_accuracy = accuracy_score(y_test, astar_predictions)
    return {
        "accuracy": bundle.get("accuracy", 0.0),
        "confusion_matrix": bundle.get("confusion_matrix"),
        "classes": bundle.get("classes", []),
        "astar_accuracy": astar_accuracy,
        "astar_mean_nodes_expanded": float(np.mean(astar_nodes)),
    }


if __name__ == "__main__":
    print("Training Random Forest...")
    metrics = train_and_save()
    print(f"Accuracy: {metrics['accuracy']}%")
    print(metrics["report"])
