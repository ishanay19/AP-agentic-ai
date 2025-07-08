import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Accounts Payable Agentic AI", page_icon="💼")

# ─── FLOATING TOGGLE CSS ────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #1e1e1e !important; color: #eee !important; }
h1,h2,h3 { color: #fff !important; }

/* Style cards, tables, expanders */
.stMetric > div, .stDataFrame, .stTable, .stExpander > div {
  background-color: #2e2e2e !important;
  color: #fff !important;
  box-shadow: 0 2px 5px rgba(0,0,0,0.5) !important;
}

/* Floating checkbox hack for the 💬 toggle */
div.stCheckbox {
  position: fixed !important;
  bottom: 20px; right: 20px;
  z-index: 1000;
}
div.stCheckbox > label > input { display: none; }           /* hide the actual box */
div.stCheckbox > label > div {
  width: 56px; height: 56px; border-radius: 28px;
  background-color: #1abc9c; color: #fff;
  font-size: 28px; display: flex;
  align-items: center; justify-content: center;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  cursor: pointer; transition: transform 0.2s ease;
}
div.stCheckbox > label:hover > div { transform: scale(1.1); }
</style>
""", unsafe_allow_html=True)

# ─── CHAT TOGGLE ─────────────────────────────────────────────────────────────
# Initialize session state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
st.checkbox("💬", key="chat_open")

# ─── MOCK DATA GENERATION ─────────────────────────────────────────────────────
@st.cache_data
def make_invoices(n=500):
    np.random.seed(42)
    vendors = ["Acme Corp","Global Supplies","Sunrise Inc.","TechSource","Alpha Traders"]
    curr    = dict(zip(vendors, ["USD","EUR","GBP","USD","EUR"]))
    cats    = ["Office Supplies","IT Equipment","Consulting","Maintenance","Logistics"]
    rows=[]; today=datetime.date.today()
    for i in range(n):
        inv = f"INV-{100000+i}"
        v   = np.random.choice(vendors)
        idt = today - datetime.timedelta(days=int(np.random.uniform(0,90)))
        pdt = idt   - datetime.timedelta(days=int(np.random.uniform(0,30)))
        amt = round(np.random.uniform(100,20000),2)
        cnt = np.random.randint(1,6)
        vals= np.random.normal(loc=amt/cnt, scale=(amt/cnt)*0.05, size=cnt)
        lsum= round(vals.sum(),2)
        rows.append({
            "invoice_id": inv, "vendor": v,
            "invoice_date": idt, "po_date": pdt,
            "amount": amt,      "line_sum": lsum,
            "currency": curr[v],
            "category": np.random.choice(cats)
        })
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["po_date"]      = pd.to_datetime(df["po_date"])
    # Dynamic metrics
    days = (df.invoice_date - df.po_date).dt.days.clip(0)
    df["date_validity"]   = (1 - days/60).clip(0,1)
    df["total_accuracy"]  = (1 - (df.amount - df.line_sum).abs()/df.amount).clip(0,1)
    df["tax_compliance"]  = np.random.uniform(0.7,1.0,len(df)).round(2)
    df["currency_match"]  = 1.0
    # Duplicates
    df["duplicate_likelihood"] = 0.0
    for v in df.vendor.unique():
        sub = df[df.vendor==v]
        for idx in sub.index:
            a   = df.at[idx,"amount"]
            cnt = (np.abs(sub.amount - a)/a < 0.01).sum() - 1
            df.at[idx,"duplicate_likelihood"] = min(cnt/5,1.0)
    # Anomaly
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"]   = -iso.fit(df[["amount"]]).decision_function(df[["amount"]])
    thr = np.percentile(df["anomaly_score"],95)
    df["amount_anomaly"]  = (df["anomaly_score"] > thr).astype(int)
    # Composite risk
    w = {"date_validity":0.2,"total_accuracy":0.3,"tax_compliance":0.2,
         "currency_match":0.1,"duplicate_likelihood":0.1,"amount_anomaly":0.1}
    df["risk_score"] = (
      df.date_validity*w["date_validity"] +
      df.total_accuracy*w["total_accuracy"] +
      df.tax_compliance*w["tax_compliance"] +
      df.currency_match*w["currency_match"] +
      (1-df.duplicate_likelihood)*w["duplicate_likelihood"] +
      (1-df.amount_anomaly)*w["amount_anomaly"]
    ) * 100
    return df.round(2)

invoices = make_invoices()

# Pre-compute charts
future = pd.date_range(datetime.date.today(), periods=30)
fc_df  = pd.DataFrame({"outflow": np.cumsum(np.random.uniform(2000,10000,30))}, index=future)

cat_counts = invoices.category.value_counts()
cat_df     = pd.DataFrame({"category":cat_counts.index, "count":cat_counts.values})
fig_cat    = px.pie(cat_df, names="category", values="count", hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Darkmint)

# ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
st.title("💼 AP Agentic AI Dashboard")

c1,c2,c3,c4 = st.columns(4, gap="large")
c1.metric("Total Invoices",      len(invoices))
c2.metric("Avg. Risk Score",     f"{invoices.risk_score.mean():.1f}%")
c3.metric("Duplicates Flagged",  int((invoices.duplicate_likelihood>0.5).sum()))
c4.metric("Anomalies Detected",  int(invoices.amount_anomaly.sum()))

st.markdown("---")
colA, colB = st.columns(2, gap="large")
with colA:
    st.subheader("📊 Spend by Category")
    st.plotly_chart(fig_cat, use_container_width=True)
with colB:
    st.subheader("🔮 Cash Flow Forecast")
    st.line_chart(fc_df, use_container_width=True)

st.markdown("---")
t1, t2 = st.columns(2, gap="large")
with t1:
    st.subheader("📈 Supplier Health")
    sup = (
        invoices.groupby("vendor")
                .agg(total_spend=("amount","sum"),
                     invoice_count=("invoice_id","count"),
                     avg_risk=("risk_score","mean"))
                .reset_index().round(1)
    )
    sup["on_time_%"] = np.random.uniform(85,100,len(sup)).round(1)
    st.dataframe(sup, height=300)
with t2:
    st.subheader("📋 Work Queue")
    q = invoices.copy()
    s = st.text_input("🔍 Search Invoice ID", "")
    if s:
        q = q[q.invoice_id.str.contains(s, case=False)]
    st.dataframe(
      q[["invoice_id","vendor","invoice_date","po_date","amount","currency","risk_score"]]
      .head(50),
      height=300
    )

st.markdown("---")
st.subheader("⚠️ Exception Resolution")
for _, r in invoices.iterrows():
    if r.risk_score<70 or r.duplicate_likelihood>0.5 or r.amount_anomaly:
        with st.expander(f"{r.invoice_id} (Risk {r.risk_score:.1f}%)"):
            st.write(f"- Duplicate Likelihood: {r.duplicate_likelihood}")
            st.write(f"- Amount Anomaly: {'Yes' if r.amount_anomaly else 'No'}")
            st.write("**AI Suggestion:** Review or auto-resolve.")

# ─── ACTION PANEL ─────────────────────────────────────────────────────────────
if "action" in st.session_state:
    act = st.session_state.action
else:
    act = None

# Display based on sidebar prompts
if st.session_state.get("action") == "high_risk":
    st.subheader("🔴 High-Risk Invoices (Risk > 90%)")
    st.dataframe(invoices[invoices.risk_score>90][["invoice_id","risk_score"]])
elif st.session_state.get("action") == "duplicates":
    st.subheader("🟠 Potential Duplicates")
    st.dataframe(invoices[invoices.duplicate_likelihood>0.5]
                 [["invoice_id","duplicate_likelihood"]])
elif st.session_state.get("action") == "cashflow":
    st.subheader("🔮 Cash Flow Forecast")
    st.line_chart(fc_df)
elif st.session_state.get("action") == "categories":
    st.subheader("📊 Category Spend Breakdown")
    st.plotly_chart(fig_cat)
elif st.session_state.get("custom_q"):
    st.subheader("💬 AI Response")
    st.write(f"Simulated action for: **{st.session_state.custom_q}**")

# ─── SIDEBAR WHEN CHAT OPEN ───────────────────────────────────────────────────
if st.session_state.chat_open:
    # Sidebar header
    st.sidebar.header("💬 Ask AP Copilot")

    # Quick prompts set session_state.action
    if st.sidebar.button("Explain High-Risk Invoices"):
        st.session_state.action = "high_risk"
    if st.sidebar.button("How duplicates are flagged?"):
        st.session_state.action = "duplicates"
    if st.sidebar.button("Show Cash Flow Forecast"):
        st.session_state.action = "cashflow"
    if st.sidebar.button("Spend Category Breakdown"):
        st.session_state.action = "categories"

    # Free-form question
    q = st.sidebar.text_input("Or ask something else…")
    if q:
        st.session_state.custom_q = q
