"""Core logic for the Finance Dashboard: parsing, categorisation, storage, analytics.

Kept separate from the Streamlit UI so it can be unit-tested.
"""
from __future__ import annotations

import io
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).with_name("finance.db")

DEFAULT_CATEGORIES = {
    "Food": ["swiggy", "zomato", "dominos", "mcdonald", "restaurant", "cafe"],
    "Groceries": ["bigbasket", "blinkit", "zepto", "dmart", "grocery", "jiomart"],
    "Transport": ["uber", "ola", "olacabs", "rapido", "irctc", "petrol", "dmrc", "metro card"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio"],
    "Bills": ["electricity", "jio", "airtel", "recharge", "broadband"],
    "Entertainment": ["netflix", "spotify", "hotstar", "bookmyshow", "prime video"],
    "Education": ["udemy", "coursera", "physics wallah"],
}

# Defaults from the first version that matched too much ("pw" in "Upwork",
# "metro" in "Metropolis"). Removed once, when an old database is migrated.
_LEGACY_BAD_DEFAULTS = [("Education", "pw"), ("Transport", "metro")]

# Payment-rail words that say nothing about the merchant.
_NOISE_TOKENS = {"upi", "imps", "neft", "rtgs", "pos", "ach", "nach", "ecs", "txn", "ref", "vpa"}

# Column-name aliases so different bank CSV formats work.
COLUMN_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date"],
    "details": ["details", "description", "narration", "particulars", "remarks"],
    "amount": ["amount", "txn amount", "transaction amount"],
    "debit": ["debit", "withdrawal", "withdrawal amt.", "withdrawal amount", "debit amount"],
    "credit": ["credit", "deposit", "deposit amt.", "deposit amount", "credit amount"],
    "type": ["debit/credit", "type", "dr/cr", "transaction type"],
}

_KEY_COLS = ["Date", "Details", "Amount", "Type"]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _find_col(columns: list[str], key: str) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in COLUMN_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    return None


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", "0")
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _add_seq(df: pd.DataFrame) -> pd.DataFrame:
    """Number identical rows (0, 1, 2...) so genuine repeats are not treated as duplicates.

    Two ₹20 chai payments on the same day are two transactions. Re-importing
    the same file still produces the same numbers, so it stays idempotent.
    """
    df = df.copy()
    df["Seq"] = df.groupby(_KEY_COLS).cumcount()
    return df


def parse_transactions(source) -> pd.DataFrame:
    """Read a bank CSV and return a clean frame: Date, Details, Amount, Type, Seq.

    Supports either (Amount + Debit/Credit column), separate Debit and
    Credit columns, or a signed Amount. Dates in most common formats are
    auto-detected (day-first for ambiguous ones like 03/04/2026).

    Rows that cannot be used are dropped, and counted in
    ``df.attrs["skipped"]`` so the UI can tell the user.
    """
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    date_col = _find_col(cols, "date")
    det_col = _find_col(cols, "details")
    if not date_col or not det_col:
        raise ValueError(
            "CSV needs a Date column and a Details/Description column. "
            f"Found columns: {cols}"
        )

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce", format="mixed")
    out["Details"] = df[det_col].astype(str).str.strip()

    amount_col = _find_col(cols, "amount")
    type_col = _find_col(cols, "type")
    debit_col = _find_col(cols, "debit")
    credit_col = _find_col(cols, "credit")

    if amount_col and type_col:
        out["Amount"] = _to_number(df[amount_col]).abs()
        t = df[type_col].astype(str).str.strip().str.lower()
        out["Type"] = t.map(
            lambda v: "Debit" if v.startswith(("d", "w")) else "Credit"
        )
    elif debit_col and credit_col:
        debit = _to_number(df[debit_col])
        credit = _to_number(df[credit_col])
        out["Amount"] = debit.where(debit > 0, credit)
        out["Type"] = debit.gt(0).map({True: "Debit", False: "Credit"})
    elif amount_col:
        signed = _to_number(df[amount_col])
        out["Amount"] = signed.abs()
        out["Type"] = signed.lt(0).map({True: "Debit", False: "Credit"})
    else:
        raise ValueError(
            "Could not find amount columns. Need Amount + Debit/Credit, "
            "or separate Debit and Credit columns."
        )

    bad_date = int(out["Date"].isna().sum())
    out = out.dropna(subset=["Date"])
    bad_amount = int((out["Amount"] <= 0).sum())
    out = out[out["Amount"] > 0].reset_index(drop=True)
    out = _add_seq(out)
    out.attrs["skipped"] = {"invalid_date": bad_date, "zero_or_invalid_amount": bad_amount}
    return out


# --------------------------------------------------------------------------- #
# Categorisation
# --------------------------------------------------------------------------- #
def normalize(text) -> str:
    """Lower-case and turn every run of punctuation into one space.

    "SWIGGY*ORDER 48213" -> "swiggy order 48213"
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def merchant_key(details: str, max_words: int = 3) -> str:
    """The reusable part of a description, used when the app learns a keyword.

    Drops payment-rail words (UPI, NEFT...) and anything containing a digit
    (reference numbers), then keeps the first few words:
    "UPI/Dr Sharma Clinic/4471829" -> "dr sharma clinic".
    """
    tokens = [
        t for t in normalize(details).split()
        if t not in _NOISE_TOKENS and not any(ch.isdigit() for ch in t)
    ]
    return " ".join(tokens[:max_words]) or normalize(details)


@lru_cache(maxsize=4096)
def _keyword_pattern(keyword: str):
    """(compiled pattern, length) for a keyword, or None if it is empty.

    A keyword must start at a word boundary, so "ola" does not match "chola".
    Short keywords (4 characters or fewer) must also END at a word boundary,
    so "ola" does not match "olive" and "cafe" does not match "cafeteria".
    Longer keywords may continue ("amazon" matches "amazonpay").
    """
    kw = normalize(keyword)
    if not kw:
        return None
    pattern = r"(?<![a-z0-9])" + re.escape(kw)
    if len(kw) <= 4:
        pattern += r"(?![a-z0-9])"
    return re.compile(pattern), len(kw)


def categorize(details: str, rules: dict[str, list[str]]) -> str:
    """Return the category whose keyword appears in `details`.

    The longest matching keyword wins, so "amazon prime" beats "amazon".
    """
    text = normalize(details)
    best_cat, best_len = "Uncategorized", 0
    for cat, keywords in rules.items():
        for kw in keywords:
            compiled = _keyword_pattern(kw)
            if compiled is None:
                continue
            pattern, length = compiled
            if length > best_len and pattern.search(text):
                best_cat, best_len = cat, length
    return best_cat


def apply_categories(df: pd.DataFrame, rules: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = df["Details"].map(lambda d: categorize(d, rules))
    return df


# --------------------------------------------------------------------------- #
# SQLite storage
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date      TEXT NOT NULL,
    details   TEXT NOT NULL,
    amount    REAL NOT NULL,
    type      TEXT NOT NULL CHECK (type IN ('Debit', 'Credit')),
    category  TEXT NOT NULL DEFAULT 'Uncategorized',
    seq       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (date, details, amount, type, seq)
);
CREATE TABLE IF NOT EXISTS category_rules (
    category TEXT NOT NULL,
    keyword  TEXT NOT NULL,
    PRIMARY KEY (category, keyword)
);
CREATE TABLE IF NOT EXISTS budgets (
    category      TEXT PRIMARY KEY,
    monthly_limit REAL NOT NULL CHECK (monthly_limit > 0)
);
CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_tx_cat  ON transactions(category);
"""


def get_conn(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    # Streamlit reruns the script on different threads, so the cached
    # connection must be usable across threads.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_add_seq(conn: sqlite3.Connection) -> None:
    """Upgrade a database created by the first version (no `seq` column).

    Existing rows are kept (seq = 0). Also removes two over-broad default
    keywords from that version.
    """
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(transactions)")]
    if not cols or "seq" in cols:
        return
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_tx_date;
        DROP INDEX IF EXISTS idx_tx_cat;
        ALTER TABLE transactions RENAME TO transactions_old;
        """
    )
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO transactions (id, date, details, amount, type, category, seq) "
        "SELECT id, date, details, amount, type, category, 0 FROM transactions_old"
    )
    conn.execute("DROP TABLE transactions_old")
    conn.executemany(
        "DELETE FROM category_rules WHERE category = ? AND keyword = ?",
        _LEGACY_BAD_DEFAULTS,
    )
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    _migrate_add_seq(conn)
    conn.executescript(_SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM category_rules").fetchone()[0] == 0:
        conn.executemany(
            "INSERT OR IGNORE INTO category_rules VALUES (?, ?)",
            [(c, k) for c, kws in DEFAULT_CATEGORIES.items() for k in kws],
        )
    conn.commit()


def load_rules(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    for r in conn.execute("SELECT category, keyword FROM category_rules ORDER BY category"):
        rules.setdefault(r["category"], []).append(r["keyword"])
    return rules


def list_categories(conn: sqlite3.Connection) -> list[str]:
    cats = {r[0] for r in conn.execute("SELECT DISTINCT category FROM category_rules")}
    cats |= {r[0] for r in conn.execute("SELECT DISTINCT category FROM transactions")}
    cats.add("Uncategorized")
    return sorted(cats)


def add_category(conn: sqlite3.Connection, category: str) -> None:
    """A category exists once it has at least one keyword; seed with its own name."""
    category = category.strip()
    if category:
        conn.execute(
            "INSERT OR IGNORE INTO category_rules VALUES (?, ?)",
            (category, category.lower()),
        )
        conn.commit()


def add_keyword(conn: sqlite3.Connection, category: str, keyword: str) -> None:
    keyword = normalize(keyword)
    if category and category != "Uncategorized" and keyword:
        conn.execute(
            "INSERT OR IGNORE INTO category_rules VALUES (?, ?)", (category, keyword)
        )
        conn.commit()


def save_transactions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert new rows, skipping ones already stored. Returns number of rows added."""
    if "Seq" not in df.columns:
        df = _add_seq(df)
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO transactions (date, details, amount, type, category, seq) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (r.Date.strftime("%Y-%m-%d"), r.Details, float(r.Amount), r.Type, r.Category, int(r.Seq))
            for r in df.itertuples(index=False)
        ],
    )
    conn.commit()
    return conn.total_changes - before


def fetch_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT id, date AS Date, details AS Details, amount AS Amount, "
        "type AS Type, category AS Category FROM transactions ORDER BY date DESC, id DESC",
        conn,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def update_category(conn: sqlite3.Connection, tx_id: int, category: str, learn: bool = True) -> str | None:
    """Set a transaction's category. Returns the keyword learned, if any."""
    row = conn.execute("SELECT details FROM transactions WHERE id = ?", (tx_id,)).fetchone()
    if row is None:
        return None
    conn.execute("UPDATE transactions SET category = ? WHERE id = ?", (category, tx_id))
    conn.commit()
    if learn and category != "Uncategorized":
        key = merchant_key(row["details"])
        add_keyword(conn, category, key)
        return key
    return None


def recategorize_all(conn: sqlite3.Connection) -> int:
    """Re-run current rules over rows that are still Uncategorized."""
    rules = load_rules(conn)
    rows = conn.execute(
        "SELECT id, details FROM transactions WHERE category = 'Uncategorized'"
    ).fetchall()
    changed = 0
    for r in rows:
        cat = categorize(r["details"], rules)
        if cat != "Uncategorized":
            conn.execute("UPDATE transactions SET category = ? WHERE id = ?", (cat, r["id"]))
            changed += 1
    conn.commit()
    return changed


def clear_transactions(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM transactions")
    conn.commit()


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #
def set_budget(conn: sqlite3.Connection, category: str, limit: float) -> None:
    conn.execute(
        "INSERT INTO budgets (category, monthly_limit) VALUES (?, ?) "
        "ON CONFLICT(category) DO UPDATE SET monthly_limit = excluded.monthly_limit",
        (category, float(limit)),
    )
    conn.commit()


def delete_budget(conn: sqlite3.Connection, category: str) -> None:
    conn.execute("DELETE FROM budgets WHERE category = ?", (category,))
    conn.commit()


def get_budgets(conn: sqlite3.Connection) -> dict[str, float]:
    return {r["category"]: r["monthly_limit"] for r in conn.execute("SELECT * FROM budgets")}


def budget_status(debits: pd.DataFrame, budgets: dict[str, float], month: str) -> pd.DataFrame:
    """Spend vs limit per budgeted category for a 'YYYY-MM' month."""
    if not budgets:
        return pd.DataFrame(columns=["Category", "Spent", "Limit", "Used %", "Status"])
    m = debits[debits["Date"].dt.strftime("%Y-%m") == month]
    spent = m.groupby("Category")["Amount"].sum()
    rows = []
    for cat, limit in budgets.items():
        s = float(spent.get(cat, 0.0))
        pct = s / limit * 100
        status = "Over budget" if pct > 100 else "Warning" if pct >= 80 else "OK"
        rows.append({"Category": cat, "Spent": s, "Limit": limit, "Used %": round(pct, 1), "Status": status})
    return pd.DataFrame(rows).sort_values("Used %", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Income, expenses and savings per month."""
    if df.empty:
        return pd.DataFrame(columns=["Month", "Income", "Expenses", "Savings"])
    tmp = df.assign(Month=df["Date"].dt.strftime("%Y-%m"))
    pivot = tmp.pivot_table(index="Month", columns="Type", values="Amount", aggfunc="sum", fill_value=0.0)
    for col in ("Credit", "Debit"):
        if col not in pivot:
            pivot[col] = 0.0
    out = pivot.rename(columns={"Credit": "Income", "Debit": "Expenses"})[["Income", "Expenses"]]
    out["Savings"] = out["Income"] - out["Expenses"]
    return out.reset_index()


def category_totals(debits: pd.DataFrame) -> pd.DataFrame:
    if debits.empty:
        return pd.DataFrame(columns=["Category", "Amount"])
    return (
        debits.groupby("Category", as_index=False)["Amount"].sum()
        .sort_values("Amount", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def build_excel_report(df: pd.DataFrame) -> bytes:
    """Multi-sheet Excel report: transactions, monthly summary, category totals."""
    debits = df[df["Type"] == "Debit"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.drop(columns=["id"], errors="ignore").to_excel(xw, sheet_name="Transactions", index=False)
        monthly_summary(df).to_excel(xw, sheet_name="Monthly Summary", index=False)
        category_totals(debits).to_excel(xw, sheet_name="By Category", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 3, 45)
    return buf.getvalue()
