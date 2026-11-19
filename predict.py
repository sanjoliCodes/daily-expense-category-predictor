"""Predict the category of a new expense using the trained model.

Usage:
    python predict.py                       # interactive mode
    python predict.py 250 Friday "pizza with friends"
"""
import sys

import joblib
import pandas as pd

MODEL_FILE = "expense_model.joblib"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def predict(model, amount, day, description):
    row = pd.DataFrame([{"amount": float(amount), "day": day.capitalize(),
                         "description": description}])
    category = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    confidence = probs.max()
    return category, confidence


def main():
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        sys.exit("Model not found. Run 'python train.py' first.")

    if len(sys.argv) >= 4:
        amount, day, description = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
        category, conf = predict(model, amount, day, description)
        print(f"Predicted category: {category} ({conf:.0%} confident)")
        return

    print("Daily Expense Category Predictor (type 'q' to quit)")
    while True:
        amount = input("\nAmount (Rs): ").strip()
        if amount.lower() == "q":
            break
        day = input(f"Day ({', '.join(DAYS)}): ").strip()
        description = input("Description: ").strip()
        try:
            category, conf = predict(model, amount, day, description)
        except ValueError:
            print("Please enter a valid number for the amount.")
            continue
        print(f"-> Predicted category: {category} ({conf:.0%} confident)")


if __name__ == "__main__":
    main()
