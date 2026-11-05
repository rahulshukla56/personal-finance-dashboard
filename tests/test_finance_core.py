import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finance_core as fc  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "sample_transactions.csv"


@pytest.fixture
def conn(tmp_path):
    c = fc.get_conn(tmp_path / "test.db")
    fc.init_db(c)
    yield c
    c.close()


def load_sample(conn):
    df = fc.apply_categories(fc.parse_transactions(SAMPLE), fc.load_rules(conn))
    fc.save_transactions(conn, df)
    return fc.fetch_transactions(conn)


# ---- parsing --------------------------------------------------------------- #
def test_parse_amount_with_type_column():
    df = fc.parse_transactions(SAMPLE)
    assert set(df["Type"]) == {"Debit", "Credit"}
    assert df["Amount"].gt(0).all()
    assert df.loc[df["Details"].str.contains("BigBasket"), "Amount"].iloc[0] == 1840.0


def test_parse_separate_debit_credit_columns():
    csv = io.StringIO(
        "Txn Date,Narration,Withdrawal Amt.,Deposit Amt.\n"
        "05/10/2026,UPI SWIGGY,250.00,\n"
        "06/10/2026,SALARY,,30000\n"
    )
    df = fc.parse_transactions(csv)
    assert list(df["Type"]) == ["Debit", "Credit"]
    assert list(df["Amount"]) == [250.0, 30000.0]
    assert df["Date"].iloc[0].month == 10  # dd/mm/yyyy


def test_parse_signed_amount_column():
    csv = io.StringIO("Date,Description,Amount\n2026-10-01,Coffee,-120\n2026-10-02,Refund,50\n")
    df = fc.parse_transactions(csv)
    assert list(df["Type"]) == ["Debit", "Credit"]


def test_parse_missing_columns_raises():
    with pytest.raises(ValueError):
        fc.parse_transactions(io.StringIO("foo,bar\n1,2\n"))


def test_skipped_rows_are_counted():
    csv = io.StringIO(
        "Date,Details,Amount,Debit/Credit\n"
        "not a date,X,10,Debit\n"
        "01 Oct 2026,Y,0,Debit\n"
        "02 Oct 2026,Z,50,Debit\n"
    )
    df = fc.parse_transactions(csv)
    assert len(df) == 1
    assert df.attrs["skipped"] == {"invalid_date": 1, "zero_or_invalid_amount": 1}


def test_clean_file_reports_nothing_skipped():
    df = fc.parse_transactions(SAMPLE)
    assert sum(df.attrs["skipped"].values()) == 0


# ---- categorisation -------------------------------------------------------- #
def test_partial_match_and_case_insensitive():
    rules = {"Food": ["swiggy"]}
    assert fc.categorize("SWIGGY*ORDER 48213", rules) == "Food"
    assert fc.categorize("Random shop", rules) == "Uncategorized"


def test_longest_keyword_wins():
    rules = {"Shopping": ["amazon"], "Entertainment": ["amazon prime"]}
    assert fc.categorize("Amazon Prime Video", rules) == "Entertainment"
    assert fc.categorize("Amazon Pay Order", rules) == "Shopping"


@pytest.mark.parametrize(
    "text",
    ["Chola Finance EMI", "Coca Cola Store", "Metropolis Lab", "Upwork Payment", "Cafeteria Block B"],
)
def test_short_keywords_do_not_match_inside_other_words(conn, text):
    assert fc.categorize(text, fc.load_rules(conn)) == "Uncategorized"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("AMAZONPAY INDIA", "Shopping"),
        ("OLA CABS BLR", "Transport"),
        ("Ola Ride 8832", "Transport"),
        ("SWIGGY*ORDER 99120", "Food"),
        ("Prime Video Subscription", "Entertainment"),
        ("UPI/Zomato Online", "Food"),
    ],
)
def test_real_looking_descriptions_still_match(conn, text, expected):
    assert fc.categorize(text, fc.load_rules(conn)) == expected


def test_merchant_key_strips_reference_numbers_and_rails():
    assert fc.merchant_key("UPI/Dr Sharma Clinic/4471829") == "dr sharma clinic"
    assert fc.merchant_key("SWIGGY*ORDER 48213") == "swiggy order"
    assert fc.merchant_key("NEFT 998877") == "neft 998877"  # nothing useful left: fall back


# ---- database -------------------------------------------------------------- #
def test_import_is_idempotent(conn):
    df = fc.apply_categories(fc.parse_transactions(SAMPLE), fc.load_rules(conn))
    first = fc.save_transactions(conn, df)
    second = fc.save_transactions(conn, df)
    assert first == len(df)
    assert second == 0


def test_identical_same_day_transactions_are_both_kept(conn):
    csv = (
        "Date,Details,Amount,Debit/Credit\n"
        "05 Oct 2026,Chai Point,20,Debit\n"
        "05 Oct 2026,Chai Point,20,Debit\n"
    )
    df = fc.apply_categories(fc.parse_transactions(io.StringIO(csv)), fc.load_rules(conn))
    assert fc.save_transactions(conn, df) == 2
    assert fc.save_transactions(conn, df) == 0  # re-import still safe
    assert fc.fetch_transactions(conn)["Amount"].sum() == 40


def test_learning_from_manual_change(conn):
    data = load_sample(conn)
    row = data[data["Details"] == "Doctor Consultation"].iloc[0]
    assert row["Category"] == "Uncategorized"
    fc.add_category(conn, "Health")
    fc.update_category(conn, int(row["id"]), "Health")
    assert "doctor consultation" in fc.load_rules(conn)["Health"]
    assert fc.categorize("Doctor Consultation", fc.load_rules(conn)) == "Health"


def test_learning_generalises_across_reference_numbers(conn):
    csv = "Date,Details,Amount,Debit/Credit\n01 Oct 2026,UPI/Dr Sharma Clinic/4471829,500,Debit\n"
    df = fc.apply_categories(fc.parse_transactions(io.StringIO(csv)), fc.load_rules(conn))
    fc.save_transactions(conn, df)
    tx_id = int(fc.fetch_transactions(conn)["id"].iloc[0])

    fc.add_category(conn, "Health")
    learned = fc.update_category(conn, tx_id, "Health")

    assert learned == "dr sharma clinic"
    # a later visit with a different reference number is now recognised
    assert fc.categorize("UPI/Dr Sharma Clinic/9920331", fc.load_rules(conn)) == "Health"


def test_recategorize_all_fills_uncategorized(conn):
    load_sample(conn)
    fc.add_keyword(conn, "Cash", "atm")
    assert fc.recategorize_all(conn) >= 1
    data = fc.fetch_transactions(conn)
    assert data.loc[data["Details"].str.contains("ATM"), "Category"].iloc[0] == "Cash"


def test_old_database_is_migrated(tmp_path):
    """A database made by the first version (no seq column) keeps its data."""
    c = fc.get_conn(tmp_path / "old.db")
    c.executescript(
        """
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL, details TEXT NOT NULL, amount REAL NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('Debit', 'Credit')),
            category TEXT NOT NULL DEFAULT 'Uncategorized',
            UNIQUE (date, details, amount, type)
        );
        CREATE INDEX idx_tx_date ON transactions(date);
        CREATE TABLE category_rules (category TEXT NOT NULL, keyword TEXT NOT NULL, PRIMARY KEY (category, keyword));
        INSERT INTO category_rules VALUES ('Education', 'pw'), ('Transport', 'metro'), ('Food', 'swiggy');
        INSERT INTO transactions (date, details, amount, type, category)
            VALUES ('2026-10-01', 'Old Row', 10, 'Debit', 'Food');
        """
    )
    fc.init_db(c)

    cols = [r["name"] for r in c.execute("PRAGMA table_info(transactions)")]
    assert "seq" in cols
    assert c.execute("SELECT details, category FROM transactions").fetchall()[0]["details"] == "Old Row"

    rules = fc.load_rules(c)
    assert "pw" not in rules.get("Education", [])
    assert "metro" not in rules.get("Transport", [])
    assert "swiggy" in rules["Food"]

    # identical rows can be stored after the upgrade
    csv = "Date,Details,Amount,Debit/Credit\n05 Oct 2026,Chai,20,Debit\n05 Oct 2026,Chai,20,Debit\n"
    df = fc.apply_categories(fc.parse_transactions(io.StringIO(csv)), rules)
    assert fc.save_transactions(c, df) == 2
    c.close()


# ---- analytics ------------------------------------------------------------- #
def test_monthly_summary(conn):
    data = load_sample(conn)
    m = fc.monthly_summary(data)
    assert list(m["Month"]) == ["2026-09", "2026-10", "2026-11"]
    sep = m[m["Month"] == "2026-09"].iloc[0]
    assert sep["Income"] == 51000.0
    assert sep["Savings"] == pytest.approx(sep["Income"] - sep["Expenses"])


def test_budget_status_levels(conn):
    data = load_sample(conn)
    debits = data[data["Type"] == "Debit"]
    fc.set_budget(conn, "Food", 800)    # Oct food = 450 + 310 = 760 -> 95%
    fc.set_budget(conn, "Shopping", 1000)  # Oct shopping = 3499 -> over
    fc.set_budget(conn, "Transport", 5000)  # Oct = 90 -> OK
    s = fc.budget_status(debits, fc.get_budgets(conn), "2026-10").set_index("Category")
    assert s.loc["Food", "Status"] == "Warning"
    assert s.loc["Shopping", "Status"] == "Over budget"
    assert s.loc["Transport", "Status"] == "OK"


def test_excel_report_has_expected_sheets(conn):
    data = load_sample(conn)
    xl = pd.ExcelFile(io.BytesIO(fc.build_excel_report(data)))
    assert xl.sheet_names == ["Transactions", "Monthly Summary", "By Category"]
