"""Train a simple classifier that predicts the category of a daily expense.

Uses: amount (number), day (category), description (text keywords via TF-IDF).
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "expenses.csv"
MODEL_FILE = "expense_model.joblib"


def build_pipeline():
    features = ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), lowercase=True), "description"),
            ("day", OneHotEncoder(handle_unknown="ignore"), ["day"]),
            ("amount", StandardScaler(), ["amount"]),
        ]
    )
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[["amount", "day", "description"]]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}\n")
    print(classification_report(y_test, preds))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Labels: {list(model.classes_)}")
    print(confusion_matrix(y_test, preds, labels=model.classes_))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
