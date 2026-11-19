"""Generate a synthetic dataset of daily student expenses -> expenses.csv

Columns: amount, day, description, category
"""
import random

import pandas as pd

random.seed(42)

# category -> (description templates, (min_amount, max_amount))
CATEGORIES = {
    "Food": (
        ["lunch at canteen", "dinner with friends", "pizza", "burger and fries",
         "tea and samosa", "breakfast paratha", "ice cream", "biryani",
         "coffee at cafe", "snacks from shop", "maggi and juice", "thali meal"],
        (30, 400),
    ),
    "Travel": (
        ["bus ticket", "auto rickshaw fare", "metro recharge", "uber ride",
         "ola cab to college", "train ticket home", "petrol for bike",
         "e-rickshaw", "bus pass monthly", "cab to railway station"],
        (20, 800),
    ),
    "Study": (
        ["notebook and pens", "textbook purchase", "photocopy notes",
         "online course fee", "exam form fee", "lab manual", "stationery",
         "library fine", "project printing", "coaching fees", "geometry box"],
        (20, 1500),
    ),
    "Entertainment": (
        ["movie ticket", "netflix subscription", "spotify premium",
         "gaming recharge", "concert pass", "bowling with friends",
         "amazon prime", "arcade games", "youtube premium", "cricket match entry"],
        (50, 1000),
    ),
    "Shopping": (
        ["new t-shirt", "sneakers", "backpack", "earphones", "phone cover",
         "jeans", "hoodie", "watch", "water bottle", "shampoo and soap"],
        (100, 3000),
    ),
    "Bills": (
        ["mobile recharge", "wifi bill", "hostel mess fee", "electricity share",
         "laundry charges", "room rent share", "data pack", "gas cylinder share"],
        (50, 3000),
    ),
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make_row(category):
    templates, (lo, hi) = CATEGORIES[category]
    description = random.choice(templates)
    amount = round(random.uniform(lo, hi) / 10) * 10  # round to nearest 10
    day = random.choice(DAYS)
    # small realism tweaks: more entertainment / food on weekends
    if category in ("Entertainment", "Food") and random.random() < 0.4:
        day = random.choice(["Saturday", "Sunday"])
    return {"amount": max(amount, 10), "day": day,
            "description": description, "category": category}


def main(rows_per_category=60):
    rows = [make_row(c) for c in CATEGORIES for _ in range(rows_per_category)]
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv("expenses.csv", index=False)
    print(f"Saved expenses.csv with {len(df)} rows")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
