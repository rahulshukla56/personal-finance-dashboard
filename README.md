# Personal Finance Dashboard

A Streamlit web app that turns a bank statement CSV into a spending dashboard: auto-categorised expenses, monthly trends, budget alerts and an Excel report. Data is stored in SQLite, and the categoriser learns from your corrections.

**Tech:** Python, pandas, SQLite, Streamlit, Plotly, openpyxl, pytest

<!-- Add screenshots here: docs/overview.png, docs/budgets.png -->

## Features

- **Flexible CSV import**: detects common column names (Date, Description/Narration/Details) and three layouts: `Amount + Debit/Credit`, separate `Debit`/`Credit` columns, or a signed `Amount`. Mixed date formats are handled.
- **Smart categorisation**: case-insensitive keyword matching on word boundaries, so `SWIGGY*ORDER 48213` matches `swiggy` and `AMAZONPAY` matches `amazon`, but `Chola Finance` does not match `ola`. The longest keyword wins (`amazon prime` beats `amazon`).
- **Learns from you**: change a category once and the merchant name is remembered. Reference numbers and payment prefixes (UPI, NEFT) are ignored, so the next `UPI/Dr Sharma Clinic/<new ref>` is categorised automatically.
- **SQLite storage** with indexes and a unique constraint, so re-importing the same file never creates duplicates, while genuinely repeated transactions (two identical payments on one day) are both kept. Databases from older versions are upgraded automatically.
- **Monthly trend** of income vs expenses, savings and savings rate.
- **Budgets with alerts**: OK / Warning (80%+) / Over budget, per category per month.
- **Excel export** with three sheets: Transactions, Monthly Summary, By Category.
- **Import feedback**: rows that cannot be read (bad date, zero amount) are counted and reported instead of silently dropped.
- **29 unit tests** for parsing, categorisation, storage, migration, budgets and export.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then upload `sample_data/sample_transactions.csv` from the sidebar and click **Import file**.

## Run tests

```bash
python -m pytest -q
```

## Project structure

```
app.py              Streamlit UI
finance_core.py     Parsing, categorisation, SQLite, analytics, export (no UI code)
tests/              pytest unit tests
sample_data/        Example bank CSV
```

## Database schema

| Table | Purpose |
|---|---|
| `transactions` | date, details, amount, type (Debit/Credit), category, seq; unique on (date, details, amount, type, seq) |
| `category_rules` | category → keyword pairs used for auto-categorisation |
| `budgets` | monthly limit per category |

## Deploy (free)

Push to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`. Note: SQLite data on Streamlit Cloud is not permanent. For permanent storage, switch the connection to PostgreSQL.

## Ideas for next steps

- User login so several people can use one deployment
- PostgreSQL instead of SQLite
- Recurring-payment detection
- ML-based categorisation (a scikit-learn text classifier trained on your corrected transactions)
