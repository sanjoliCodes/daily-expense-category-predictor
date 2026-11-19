# Daily Expense Category Predictor

A beginner text/tabular classification project for personal expense management.

Students record many expenses and may find it difficult to classify them consistently.
This project uses the **amount**, **day** and **description** keywords of an expense to
predict a category such as Food, Travel, Study, Entertainment, Shopping or Bills.

## How it works
- `description` -> converted to numbers with TF-IDF (keyword features)
- `day` -> one-hot encoded
- `amount` -> standardised
- A Logistic Regression classifier is trained on these features.

## Project files
| File | Purpose |
|------|---------|
| `generate_data.py` | Creates the sample dataset `expenses.csv` |
| `train.py` | Trains the model, prints accuracy, saves `expense_model.joblib` |
| `predict.py` | Predicts the category of a new expense |
| `requirements.txt` | Python libraries needed |

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python generate_data.py     # 1. create dataset
python train.py             # 2. train + evaluate model
python predict.py           # 3. interactive prediction
python predict.py 250 Friday "pizza with friends"
```

## Author
Sanjoli (Roll No. 124664, Batch B1, 2026)
