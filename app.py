import pandas as pd
import plotly.express as px
import streamlit as st

import finance_core as fc

st.set_page_config(page_title="Personal Finance Dashboard", page_icon="💰", layout="wide")


@st.cache_resource
def get_connection():
    conn = fc.get_conn()
    fc.init_db(conn)
    return conn


conn = get_connection()


def money(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.2f}"


# ----------------------------- Sidebar ------------------------------------- #
with st.sidebar:
    st.header("Setup")
    symbol = st.selectbox("Currency", ["₹", "$", "€", "£", "AED "], index=0)

    st.subheader("Import transactions")
    uploaded = st.file_uploader("Bank statement CSV", type=["csv"])
    if uploaded is not None and st.button("Import file", type="primary"):
        try:
            parsed = fc.parse_transactions(uploaded)
            skipped = parsed.attrs.get("skipped", {})
            categorized = fc.apply_categories(parsed, fc.load_rules(conn))
            added = fc.save_transactions(conn, categorized)
            already = len(categorized) - added
            st.success(f"Imported {added} new transactions ({already} already in the database).")

            n_bad = sum(skipped.values())
            if n_bad:
                st.warning(
                    f"Skipped {n_bad} row(s) from the file: "
                    f"{skipped.get('invalid_date', 0)} with an unreadable date, "
                    f"{skipped.get('zero_or_invalid_amount', 0)} with a zero or unreadable amount."
                )
        except Exception as e:  # noqa: BLE001 - show any parsing problem to the user
            st.error(f"Could not read file: {e}")

    st.caption(
        "Accepted columns: Date, Details/Description/Narration, and either "
        "Amount + Debit/Credit, or separate Debit and Credit columns."
    )

    with st.expander("Danger zone"):
        if st.button("Delete all transactions"):
            fc.clear_transactions(conn)
            st.rerun()

# ----------------------------- Data ---------------------------------------- #
data = fc.fetch_transactions(conn)

st.title("💰 Personal Finance Dashboard")

if data.empty:
    st.info("No data yet. Upload a bank CSV in the sidebar (a sample is in `sample_data/`).")
    st.stop()

months = sorted(data["Date"].dt.strftime("%Y-%m").unique(), reverse=True)
with st.sidebar:
    st.subheader("Filter")
    month_choice = st.selectbox("Month", ["All time"] + months)

view = data if month_choice == "All time" else data[data["Date"].dt.strftime("%Y-%m") == month_choice]
debits = view[view["Type"] == "Debit"]
credits = view[view["Type"] == "Credit"]

income = credits["Amount"].sum()
expenses = debits["Amount"].sum()
savings = income - expenses
rate = (savings / income * 100) if income else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Income", money(income, symbol))
c2.metric("Expenses", money(expenses, symbol))
c3.metric("Savings", money(savings, symbol))
c4.metric("Savings rate", f"{rate:.1f}%")

tab_over, tab_tx, tab_budget, tab_income = st.tabs(
    ["Overview", "Transactions", "Budgets", "Income"]
)

# ----------------------------- Overview ------------------------------------ #
with tab_over:
    left, right = st.columns(2)

    totals = fc.category_totals(debits)
    with left:
        st.subheader("Expenses by category")
        if totals.empty:
            st.write("No expenses in this period.")
        else:
            st.plotly_chart(
                px.pie(totals, values="Amount", names="Category", hole=0.4),
                width="stretch",
            )
    with right:
        st.subheader("Top categories")
        if not totals.empty:
            st.dataframe(
                totals,
                column_config={"Amount": st.column_config.NumberColumn("Amount", format=f"{symbol}%.2f")},
                hide_index=True,
                width="stretch",
            )

    st.subheader("Monthly trend")
    monthly = fc.monthly_summary(data)
    if not monthly.empty:
        long = monthly.melt(id_vars="Month", value_vars=["Income", "Expenses"], var_name="Type", value_name="Amount")
        fig = px.bar(long, x="Month", y="Amount", color="Type", barmode="group")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(monthly, hide_index=True, width="stretch")

    st.download_button(
        "Download Excel report",
        data=fc.build_excel_report(view),
        file_name="finance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ----------------------------- Transactions -------------------------------- #
with tab_tx:
    st.subheader("Edit categories")
    st.caption(
        "Change a category and click Save. The app remembers the merchant name "
        "(reference numbers are ignored), so similar transactions are categorised next time."
    )

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_cat = st.text_input("New category name", key="new_cat")
    with col_b:
        st.write("")
        st.write("")
        if st.button("Add category") and new_cat.strip():
            fc.add_category(conn, new_cat)
            st.rerun()

    options = fc.list_categories(conn)
    only_uncat = st.checkbox("Show only Uncategorized")
    table = debits if not only_uncat else debits[debits["Category"] == "Uncategorized"]

    edited = st.data_editor(
        table[["id", "Date", "Details", "Amount", "Category"]],
        column_config={
            "id": None,
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY", disabled=True),
            "Details": st.column_config.TextColumn("Details", disabled=True),
            "Amount": st.column_config.NumberColumn("Amount", format=f"{symbol}%.2f", disabled=True),
            "Category": st.column_config.SelectboxColumn("Category", options=options, required=True),
        },
        hide_index=True,
        width="stretch",
        key="editor",
    )

    if st.button("Save changes", type="primary"):
        original = table.set_index("id")["Category"]
        learned = []
        changes = 0
        for row in edited.itertuples(index=False):
            if original.get(row.id) != row.Category:
                key = fc.update_category(conn, int(row.id), row.Category)
                changes += 1
                if key:
                    learned.append(f"'{key}' → {row.Category}")
        if changes:
            auto = fc.recategorize_all(conn)
            msg = f"Updated {changes} transaction(s)."
            if learned:
                msg += " Learned: " + ", ".join(sorted(set(learned))) + "."
            if auto:
                msg += f" {auto} other uncategorized transaction(s) were categorized automatically."
            st.session_state["flash"] = msg
            st.rerun()
        else:
            st.info("No changes to save.")

# ----------------------------- Budgets ------------------------------------- #
with tab_budget:
    st.subheader("Monthly budgets")
    budget_month = month_choice if month_choice != "All time" else months[0]
    st.caption(f"Showing spend for {budget_month}. Pick a month in the sidebar to change it.")

    with st.form("budget_form", clear_on_submit=True):
        b1, b2 = st.columns(2)
        b_cat = b1.selectbox("Category", [c for c in options if c != "Uncategorized"])
        b_lim = b2.number_input("Monthly limit", min_value=1.0, value=5000.0, step=500.0)
        if st.form_submit_button("Save budget"):
            fc.set_budget(conn, b_cat, b_lim)
            st.rerun()

    status = fc.budget_status(data[data["Type"] == "Debit"], fc.get_budgets(conn), budget_month)
    if status.empty:
        st.write("No budgets set yet.")
    else:
        for r in status.itertuples():
            label = f"{r.Category}: {money(r.Spent, symbol)} of {money(r.Limit, symbol)} ({r._4}%)"
            if r.Status == "Over budget":
                st.error(f"🚨 {label}  Over budget")
            elif r.Status == "Warning":
                st.warning(f"⚠️ {label}  Close to the limit")
            else:
                st.success(f"✅ {label}")
            st.progress(min(r._4 / 100, 1.0))
        to_remove = st.selectbox("Remove a budget", [""] + list(status["Category"]))
        if to_remove and st.button("Remove"):
            fc.delete_budget(conn, to_remove)
            st.rerun()

# ----------------------------- Income -------------------------------------- #
with tab_income:
    st.subheader("Income / payments received")
    st.metric("Total received", money(income, symbol))
    st.dataframe(
        credits[["Date", "Details", "Amount"]],
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "Amount": st.column_config.NumberColumn("Amount", format=f"{symbol}%.2f"),
        },
        hide_index=True,
        width="stretch",
    )
