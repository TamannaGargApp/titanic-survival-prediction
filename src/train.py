"""
Titanic Survival Prediction - preprocessing, model comparison, final Random Forest.

Script version of notebooks/titanic_preprocessing_and_model.ipynb.

Usage:
    python src/train.py                       # reads data/train.csv
    python src/train.py --data path/to/train.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "models" / "titanic_random_forest.joblib"

DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Data file not found: {path}\n"
            "Download train.csv from Kaggle and put it in the data/ folder.\n"
            "See README.md -> 'Getting the data'."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Titanic data and turn it into numeric features."""
    df = df.copy()

    # 1. Drop IDs / free text and Cabin (~77% missing)
    df = df.drop(columns=DROP_COLUMNS)

    # 2. Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())                # 177 missing
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # 2 missing

    # 3. Encode categorical columns as numbers
    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["Embarked"] = df["Embarked"].map(EMBARKED_MAP)

    return df


def build_models() -> dict:
    return {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=500),
        "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Titanic survival model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to train.csv")
    args = parser.parse_args()

    df = preprocess(load_data(args.data))
    X = df.drop(columns="Survived")
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    # Compare models on the hold-out test set AND with 5-fold cross-validation
    print(f"{'Model':<32}{'Test acc':>10}{'5-fold CV':>18}")
    print("-" * 60)
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cv = cross_val_score(model, X, y, cv=5)
        print(f"{name:<32}{test_acc:>10.3f}{cv.mean():>11.3f} ± {cv.std():.3f}")

    # Final model: Random Forest
    final_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print(f"\nFinal model: Random Forest  |  test accuracy {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFeature importance:")
    importance = pd.Series(final_model.feature_importances_, index=X.columns)
    for col, score in importance.sort_values(ascending=False).items():
        print(f"  {col:<9} {score:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
