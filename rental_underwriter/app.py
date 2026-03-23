import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Rental Property Underwriter",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEALS_FILE = Path(__file__).parent / "saved_deals.json"

# ─── persistence helpers ────────────────────────────────────────────────────

def load_all_deals() -> dict:
    if DEALS_FILE.exists():
        with open(DEALS_FILE) as f:
            return json.load(f)
    return {}

def save_all_deals(deals: dict):
    with open(DEALS_FILE, "w") as f:
        json.dump(deals, f, indent=2)

# ─── financial engine ───────────────────────────────────────────────────────

def calculate_monthly_mortgage(principal, annual_rate, years):
    if annual_rate == 0:
        return principal / (years * 12)
    r = annual_rate / 100 / 12
    n = years * 12
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def loan_balance(principal, annual_rate, years, payments_made):
    if annual_rate == 0:
        return max(0, principal - principal / (years * 12) * payments_made)
    r = annual_rate / 100 / 12
    n = years * 12
    return principal * ((1 + r) ** n - (1 + r) ** payments_made) / ((1 + r) ** n - 1)

def run_model(p: dict) -> dict:
    # ── loan ──────────────────────────────────────────────────────────────
    down = p["purchase_price"] * p["down_pct"] / 100
    loan = p["purchase_price"] - down
    monthly_payment = calculate_monthly_mortgage(loan, p["interest_rate"], p["loan_term"])

    # ── depreciation ──────────────────────────────────────────────────────
    depreciable_value = p["purchase_price"] * (1 - p["land_pct"] / 100)
    annual_depreciation = depreciable_value / p["dep_years"] if p["dep_enabled"] else 0

    # ── year-by-year projection ───────────────────────────────────────────
    years = p["holding_period"]
    current_loan = loan
    current_rate = p["interest_rate"]
    current_term_remaining = p["loan_term"]
    current_payment = monthly_payment
    refi_done = False
    cash_from_refi = 0.0

    cash_flows = [-down]  # year 0 = equity invested
    year_data = []

    for yr in range(1, years + 1):
        rent_income = (p["monthly_rent"] + p["other_monthly_income"]) * 12 * (
            (1 + p["rent_growth"] / 100) ** (yr - 1)
        )
        vacancy_loss = rent_income * p["vacancy_rate"] / 100
        eff_income = rent_income - vacancy_loss

        base_monthly_rent = p["monthly_rent"] * ((1 + p["rent_growth"] / 100) ** (yr - 1))
        maintenance = base_monthly_rent * 12 * p["maint_pct"] / 100
        capex = base_monthly_rent * 12 * p["capex_pct"] / 100
        prop_mgmt = base_monthly_rent * 12 * p["mgmt_pct"] / 100
        hoa = p["hoa_monthly"] * 12
        utilities = p["utilities_monthly"] * 12
        other_exp = p["other_monthly_exp"] * 12

        exp_growth = (1 + p["exp_growth"] / 100) ** (yr - 1)
        prop_tax = p["prop_taxes"] * exp_growth
        insurance = p["insurance"] * exp_growth
        hoa *= exp_growth
        utilities *= exp_growth
        other_exp *= exp_growth

        total_opex = (maintenance + capex + prop_mgmt + hoa + utilities +
                      other_exp + prop_tax + insurance)
        noi = eff_income - total_opex

        # ── refinance ─────────────────────────────────────────────────────
        if p["refi_enabled"] and yr == p["refi_year"] and not refi_done:
            prop_value_at_refi = p["purchase_price"] * (1 + p["appr_rate"] / 100) ** yr
            new_loan_amount = prop_value_at_refi * p["refi_ltv"] / 100
            refi_costs = new_loan_amount * p["refi_closing_pct"] / 100
            cash_from_refi = new_loan_amount - current_loan - refi_costs
            current_loan = new_loan_amount
            current_rate = p["refi_rate"]
            current_term_remaining = p["refi_term"]
            current_payment = calculate_monthly_mortgage(
                new_loan_amount, p["refi_rate"], p["refi_term"]
            )
            refi_done = True

        annual_debt_service = current_payment * 12
        pre_tax_cf = noi - annual_debt_service

        # ── tax benefit ───────────────────────────────────────────────────
        # interest for the year
        interest_paid = sum(
            current_loan * (current_rate / 100 / 12)
            * (1 - (1 + current_rate / 100 / 12) ** (-(current_term_remaining * 12 - m)))
            / (1 - (1 + current_rate / 100 / 12) ** (-current_term_remaining * 12))
            for m in range(12)
        ) if current_rate > 0 else 0
        taxable_income = noi - interest_paid - annual_depreciation
        tax_benefit = -taxable_income * p["tax_rate"] / 100  # negative = tax due
        after_tax_cf = pre_tax_cf + tax_benefit

        # advance loan balance (12 payments)
        for _ in range(12):
            if current_loan > 0 and current_payment > 0:
                r = current_rate / 100 / 12
                interest = current_loan * r
                principal_paid = current_payment - interest
                current_loan = max(0, current_loan - principal_paid)

        # property value
        prop_value = p["purchase_price"] * (1 + p["appr_rate"] / 100) ** yr
        equity = prop_value - current_loan

        year_data.append({
            "year": yr,
            "prop_value": prop_value,
            "loan_balance": current_loan,
            "equity": equity,
            "noi": noi,
            "annual_cf": pre_tax_cf,
            "after_tax_cf": after_tax_cf,
            "eff_income": eff_income,
            "total_opex": total_opex,
            "annual_debt_service": annual_debt_service,
        })

        cash_flows.append(pre_tax_cf)

    # ── exit ──────────────────────────────────────────────────────────────
    final = year_data[-1]
    gross_sale = final["prop_value"]
    selling_costs = gross_sale * p["selling_costs_pct"] / 100
    net_sale = gross_sale - selling_costs - final["loan_balance"]
    total_invested = down
    net_profit = net_sale - total_invested + sum(d["annual_cf"] for d in year_data)

    # IRR: replace last CF with last CF + net_sale proceeds
    irr_cfs = cash_flows[:]
    irr_cfs[-1] += net_sale
    try:
        irr = npf.irr(irr_cfs) * 100
    except Exception:
        irr = float("nan")

    # after-tax IRR
    at_cfs = [-down] + [d["after_tax_cf"] for d in year_data]
    at_cfs[-1] += net_sale
    try:
        at_irr = npf.irr(at_cfs) * 100
    except Exception:
        at_irr = float("nan")

    # ── summary metrics ───────────────────────────────────────────────────
    cap_rate = year_data[0]["noi"] / p["purchase_price"] * 100
    coc = year_data[0]["annual_cf"] / down * 100 if down > 0 else 0

    cumulative_cf = np.cumsum([d["annual_cf"] for d in year_data])
    cumulative_profit = cumulative_cf + np.array(
        [d["equity"] - (p["purchase_price"] - loan) for d in year_data]
    )

    return {
        "loan": loan,
        "monthly_payment": monthly_payment,
        "monthly_cf": year_data[0]["annual_cf"] / 12,
        "annual_cf": year_data[0]["annual_cf"],
        "noi": year_data[0]["noi"],
        "cap_rate": cap_rate,
        "coc": coc,
        "irr": irr,
        "at_irr": at_irr,
        "equity_at_exit": final["equity"],
        "net_profit": net_profit,
        "cash_from_refi": cash_from_refi,
        "year_data": year_data,
        "cumulative_profit": cumulative_profit.tolist(),
        "down": down,
    }

# ─── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #4f8ef7;
    }
    .metric-card.positive { border-left-color: #00c853; }
    .metric-card.negative { border-left-color: #ff5252; }
    .metric-card.neutral  { border-left-color: #ffd600; }
    .metric-label { font-size: 0.75rem; color: #9aa0b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #e8eaf6; }
    .metric-sub   { font-size: 0.7rem; color: #6b7399; margin-top: 2px; }
    .section-header {
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #4f8ef7;
        border-bottom: 1px solid #2e3250; padding-bottom: 4px; margin: 12px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── helpers ────────────────────────────────────────────────────────────────

def fmt_usd(v, decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{decimals}f}"

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}%"

def metric_card(label, value, sub="", positive=None):
    cls = ""
    if positive is True:
        cls = "positive"
    elif positive is False:
        cls = "negative"
    else:
        cls = "neutral"
    st.markdown(f"""
    <div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─── default inputs ──────────────────────────────────────────────────────────

DEFAULTS = dict(
    deal_name="My Deal",
    purchase_price=350000,
    down_pct=20.0,
    interest_rate=7.0,
    loan_term=30,
    monthly_rent=2500,
    other_monthly_income=0,
    vacancy_rate=5.0,
    prop_taxes=3500,
    insurance=1200,
    maint_pct=5.0,
    capex_pct=5.0,
    mgmt_pct=8.0,
    hoa_monthly=0,
    utilities_monthly=0,
    other_monthly_exp=0,
    appr_rate=3.0,
    holding_period=10,
    selling_costs_pct=6.0,
    rent_growth=2.0,
    exp_growth=2.0,
    refi_enabled=False,
    refi_year=5,
    refi_rate=6.0,
    refi_ltv=75.0,
    refi_term=30,
    refi_closing_pct=2.0,
    tax_rate=28.0,
    land_pct=20.0,
    dep_enabled=True,
    dep_years=27.5,
)

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── sidebar inputs ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏠 Rental Underwriter")

    # ── Deal management ───────────────────────────────────────────────────
    section("Deal Management")
    col_s, col_l = st.columns(2)
    with col_s:
        save_name = st.text_input("Save as", value=st.session_state.deal_name, key="save_name_input")
        if st.button("💾 Save", use_container_width=True):
            deals = load_all_deals()
            snapshot = {k: st.session_state[k] for k in DEFAULTS}
            snapshot["deal_name"] = save_name
            deals[save_name] = snapshot
            save_all_deals(deals)
            st.success(f"Saved '{save_name}'")
    with col_l:
        deals = load_all_deals()
        deal_names = list(deals.keys())
        if deal_names:
            selected = st.selectbox("Load deal", ["— select —"] + deal_names, key="load_select")
            if st.button("📂 Load", use_container_width=True):
                if selected != "— select —":
                    for k, v in deals[selected].items():
                        st.session_state[k] = v
                    st.rerun()
        else:
            st.caption("No saved deals yet")

    # ── Compare ───────────────────────────────────────────────────────────
    if deal_names:
        with st.expander("📊 Compare Deals"):
            to_compare = st.multiselect("Select deals", deal_names)
            if to_compare and st.button("Run Comparison"):
                st.session_state["compare_deals"] = to_compare
            elif not to_compare:
                st.session_state.pop("compare_deals", None)

    st.divider()

    # ── Quick Inputs ──────────────────────────────────────────────────────
    section("Quick Inputs")
    st.session_state.deal_name = st.text_input(
        "Deal Name", value=st.session_state.deal_name,
        help="A name to identify this investment scenario or property. Used only for labeling — it doesn't affect any calculations.")
    st.session_state.purchase_price = st.number_input(
        "Purchase Price ($)", value=st.session_state.purchase_price, step=1000, min_value=0,
        help="The total acquisition price of the property before financing. This is the basis for your loan amount, cap rate, and depreciation calculations.")
    if st.session_state.purchase_price >= 1000:
        st.caption(f"= ${st.session_state.purchase_price:,.0f}")
    c1, c2 = st.columns(2)
    st.session_state.down_pct = c1.number_input(
        "Down Payment %", value=st.session_state.down_pct, step=0.5, min_value=0.0, max_value=100.0,
        help="The percentage of the purchase price paid in cash at closing. This determines your loan amount and initial equity — lower down payments increase leverage but also increase monthly debt service.")
    st.session_state.interest_rate = c2.number_input(
        "Interest Rate %", value=st.session_state.interest_rate, step=0.125,
        help="The annual interest rate on the mortgage used to finance the property. Used to calculate your monthly payment and the interest portion of each payment for tax purposes.")
    c3, c4 = st.columns(2)
    st.session_state.loan_term = c3.number_input(
        "Loan Term (yrs)", value=st.session_state.loan_term, step=1, min_value=1,
        help="The amortization length of the loan in years. Longer terms lower monthly payments but result in slower equity paydown and more interest paid over time.")
    st.session_state.vacancy_rate = c4.number_input(
        "Vacancy Rate %", value=st.session_state.vacancy_rate, step=0.5, min_value=0.0, max_value=100.0,
        help="The percentage of time the property is expected to be vacant or between tenants. Applied as a haircut to gross rental income; 5–8% is typical for single-family rentals.")
    st.session_state.monthly_rent = st.number_input(
        "Monthly Rent ($)", value=st.session_state.monthly_rent, step=50, min_value=0,
        help="The expected total monthly rental income from the property before any vacancy or expense deductions. Use market rent if the property is not yet leased.")
    st.session_state.other_monthly_income = st.number_input(
        "Other Monthly Income ($)", value=st.session_state.other_monthly_income, step=25, min_value=0,
        help="Additional monthly income streams beyond base rent — e.g., parking fees, laundry, storage, or pet rent. Also subject to the vacancy haircut in the model.")

    section("Annual Expenses")
    c5, c6 = st.columns(2)
    st.session_state.prop_taxes = c5.number_input(
        "Property Taxes ($)", value=st.session_state.prop_taxes, step=100, min_value=0,
        help="Annual property taxes paid to the local government. Enter the current assessed amount; the model grows this each year at the expense growth rate.")
    if st.session_state.prop_taxes >= 1000:
        c5.caption(f"= ${st.session_state.prop_taxes:,.0f}")
    st.session_state.insurance = c6.number_input(
        "Insurance ($)", value=st.session_state.insurance, step=50, min_value=0,
        help="Annual property and liability insurance premium. This typically includes dwelling, liability, and loss-of-rent coverage. Also grown at the expense growth rate annually.")
    if st.session_state.insurance >= 1000:
        c6.caption(f"= ${st.session_state.insurance:,.0f}")

    section("% of Monthly Rent")
    c7, c8, c9 = st.columns(3)
    st.session_state.maint_pct = c7.number_input(
        "Maint %", value=st.session_state.maint_pct, step=0.5, min_value=0.0,
        help="Estimated annual maintenance and repairs as a percentage of monthly rent. Covers routine wear-and-tear items like plumbing fixes, paint, and landscaping. 5–10% is a common rule of thumb.")
    st.session_state.capex_pct = c8.number_input(
        "CapEx %", value=st.session_state.capex_pct, step=0.5, min_value=0.0,
        help="Annual reserve for capital expenditures — large future replacements like roof, HVAC, water heater, or appliances — expressed as a percentage of monthly rent. Often 5–10% depending on property age.")
    st.session_state.mgmt_pct = c9.number_input(
        "Mgmt %", value=st.session_state.mgmt_pct, step=0.5, min_value=0.0,
        help="Property management fee as a percentage of collected rent. Typically 8–12% for full-service management. Enter 0 if self-managing, but consider including an implicit cost for your time.")

    section("Other Monthly Expenses")
    c10, c11, c12 = st.columns(3)
    st.session_state.hoa_monthly = c10.number_input(
        "HOA ($)", value=st.session_state.hoa_monthly, step=10, min_value=0,
        help="Monthly HOA dues if the property is part of a homeowners or condo association. Enter the current monthly amount; grown annually at the expense growth rate.")
    st.session_state.utilities_monthly = c11.number_input(
        "Utilities ($)", value=st.session_state.utilities_monthly, step=10, min_value=0,
        help="Monthly utilities paid by the owner rather than the tenant — such as water, trash, or common-area electricity. Common in multi-unit or master-metered properties.")
    st.session_state.other_monthly_exp = c12.number_input(
        "Other ($)", value=st.session_state.other_monthly_exp, step=10, min_value=0,
        help="Any additional monthly operating expenses not captured above — e.g., lawn care, pest control, accounting fees, or a home warranty. Grown at the expense growth rate.")

    section("Exit Assumptions")
    c13, c14, c15 = st.columns(3)
    st.session_state.appr_rate = c13.number_input(
        "Appr %", value=st.session_state.appr_rate, step=0.25,
        help="Expected annual property value appreciation rate used for long-term projections. Drives the exit sale price, equity growth, and IRR. Historical US averages are roughly 3–4% annually.")
    st.session_state.holding_period = c14.number_input(
        "Hold (yrs)", value=st.session_state.holding_period, step=1, min_value=1, max_value=50,
        help="Number of years you plan to hold the property before selling. Determines the projection horizon for all charts and the year in which equity, profit, and IRR are calculated at exit.")
    st.session_state.selling_costs_pct = c15.number_input(
        "Sell Cost %", value=st.session_state.selling_costs_pct, step=0.5,
        help="Estimated total transaction costs when selling the property, as a percentage of the sale price. Typically 6–8%, covering agent commissions, title, transfer taxes, and miscellaneous closing costs.")

    # ── Advanced Assumptions ──────────────────────────────────────────────
    with st.expander("⚙️ Advanced Assumptions"):
        section("Growth Rates")
        c16, c17 = st.columns(2)
        st.session_state.rent_growth = c16.number_input(
            "Rent Growth %", value=st.session_state.rent_growth, step=0.25,
            help="Annual percentage increase in rental income applied to each projection year. Reflects expected rent escalation through lease renewals or market appreciation. 2–3% is a common baseline assumption.")
        st.session_state.exp_growth = c17.number_input(
            "Exp Growth %", value=st.session_state.exp_growth, step=0.25,
            help="Annual percentage increase applied to most operating expenses (taxes, insurance, HOA, utilities, other). Reflects inflation and cost-of-living increases. Usually modeled at 2–3% per year.")

    # ── Refinance ─────────────────────────────────────────────────────────
    with st.expander("🔄 Refinance"):
        st.session_state.refi_enabled = st.toggle(
            "Enable Refinance", value=st.session_state.refi_enabled,
            help="Model a cash-out refinance during the holding period. When enabled, the loan is replaced at the specified year using the new rate, LTV, and term — and any cash extracted is tracked separately.")
        if st.session_state.refi_enabled:
            c18, c19 = st.columns(2)
            st.session_state.refi_year = c18.number_input(
                "Refi Year", value=st.session_state.refi_year, step=1, min_value=1,
                help="The year in the holding period when the refinance occurs. At this point the property is reappraised, a new loan is originated, and any cash proceeds above the old balance (minus closing costs) are distributed.")
            st.session_state.refi_rate = c19.number_input(
                "New Rate %", value=st.session_state.refi_rate, step=0.125,
                help="The interest rate on the new refinanced loan. If you expect rates to drop, a lower rate here will also reduce your post-refi monthly payment and improve cash flow.")
            c20, c21, c22 = st.columns(3)
            st.session_state.refi_ltv = c20.number_input(
                "New LTV %", value=st.session_state.refi_ltv, step=1.0,
                help="Loan-to-value percentage for the new refinanced loan, based on the property's appraised value at the time of refinance. Higher LTV extracts more cash but increases debt service.")
            st.session_state.refi_term = c21.number_input(
                "New Term (yrs)", value=st.session_state.refi_term, step=1, min_value=1,
                help="The amortization term of the new refinanced loan in years. Resetting to a 30-year term will lower monthly payments but restarts the amortization clock, slowing equity paydown.")
            st.session_state.refi_closing_pct = c22.number_input(
                "Closing Costs %", value=st.session_state.refi_closing_pct, step=0.25,
                help="Refinance closing costs as a percentage of the new loan amount. Typically 1–3%, covering origination fees, title, appraisal, and recording. Deducted from the cash-out proceeds.")

    # ── Tax ───────────────────────────────────────────────────────────────
    with st.expander("🧾 Tax"):
        c23, c24 = st.columns(2)
        st.session_state.tax_rate = c23.number_input(
            "Marginal Tax %", value=st.session_state.tax_rate, step=1.0,
            help="Your effective marginal income tax rate, used to calculate the after-tax benefit of depreciation deductions and the tax impact on net rental income. Combine federal and state rates for accuracy.")
        st.session_state.land_pct = c24.number_input(
            "Land Value %", value=st.session_state.land_pct, step=1.0,
            help="The percentage of the purchase price attributable to land value. Land cannot be depreciated — only the structure can. Typically 15–30% for single-family homes; higher in urban markets.")
        c25, c26 = st.columns(2)
        st.session_state.dep_enabled = c25.toggle(
            "Depreciation", value=st.session_state.dep_enabled,
            help="Enable residential depreciation to model the annual non-cash tax deduction. The IRS allows straight-line depreciation of the structure (not land) over 27.5 years, which can significantly improve after-tax returns.")
        st.session_state.dep_years = c26.number_input(
            "Dep Years", value=st.session_state.dep_years, step=0.5, min_value=1.0,
            help="The depreciation schedule in years. Residential rental property uses 27.5 years (IRS standard). Commercial property uses 39 years. Shorter schedules increase the annual deduction but are only applicable in special circumstances.")

# ─── gather params ───────────────────────────────────────────────────────────

params = {k: st.session_state[k] for k in DEFAULTS}

# ─── run model ───────────────────────────────────────────────────────────────

res = run_model(params)
df  = pd.DataFrame(res["year_data"])

# ─── main panel ──────────────────────────────────────────────────────────────

st.markdown(f"## 🏠 {params['deal_name']}")
st.markdown(f"""
<div style="display:flex; gap:36px; align-items:flex-start; flex-wrap:wrap;
            margin:-4px 0 20px 0; padding:14px 18px; background:#1a1d2e;
            border-radius:8px; border:1px solid #2e3250;">
  <div>
    <div style="font-size:0.68rem;color:#9aa0b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">Purchase Price</div>
    <div style="font-size:1.3rem;font-weight:700;color:#e8eaf6;line-height:1;">{fmt_usd(params['purchase_price'])}</div>
  </div>
  <div style="color:#2e3250;font-size:1.4rem;align-self:center;">|</div>
  <div>
    <div style="font-size:0.68rem;color:#9aa0b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">Down Payment</div>
    <div style="font-size:1.3rem;font-weight:700;color:#e8eaf6;line-height:1;">{fmt_usd(res['down'])} <span style="font-size:0.9rem;font-weight:500;color:#6b7399;">({params['down_pct']:.0f}%)</span></div>
  </div>
  <div style="color:#2e3250;font-size:1.4rem;align-self:center;">|</div>
  <div>
    <div style="font-size:0.68rem;color:#9aa0b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">Loan Amount</div>
    <div style="font-size:1.3rem;font-weight:700;color:#e8eaf6;line-height:1;">{fmt_usd(res['loan'])}</div>
  </div>
  <div style="color:#2e3250;font-size:1.4rem;align-self:center;">|</div>
  <div>
    <div style="font-size:0.68rem;color:#9aa0b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">Monthly Payment</div>
    <div style="font-size:1.3rem;font-weight:700;color:#e8eaf6;line-height:1;">{fmt_usd(res['monthly_payment'])}<span style="font-size:0.85rem;font-weight:400;color:#6b7399;">/mo</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── key metrics ───────────────────────────────────────────────────────────────
st.markdown("### Key Metrics")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    metric_card("Monthly Cash Flow",  fmt_usd(res["monthly_cf"]),
                "after all expenses & debt", positive=res["monthly_cf"] >= 0)
    metric_card("Annual Cash Flow",   fmt_usd(res["annual_cf"]),
                "year 1", positive=res["annual_cf"] >= 0)
    metric_card("Cash-on-Cash Return", fmt_pct(res["coc"]),
                "year 1 pre-tax", positive=res["coc"] >= 6)

with col_b:
    metric_card("Cap Rate",   fmt_pct(res["cap_rate"]),  "year 1 NOI / purchase price", positive=res["cap_rate"] >= 6)
    metric_card("NOI",        fmt_usd(res["noi"]),       "year 1 net operating income")
    metric_card("Loan Amount", fmt_usd(res["loan"]),     f"{params['loan_term']}-year @ {params['interest_rate']}%", positive=None)

with col_c:
    metric_card("IRR",         fmt_pct(res["irr"]),    "unlevered total return", positive=res["irr"] >= 10)
    metric_card("After-Tax IRR", fmt_pct(res["at_irr"]), "with depreciation & tax", positive=res["at_irr"] >= 8)
    metric_card("Equity at Exit", fmt_usd(res["equity_at_exit"]),
                f"year {params['holding_period']}", positive=True)

with col_d:
    metric_card("Net Profit at Exit", fmt_usd(res["net_profit"]),
                "equity + cum. cash flow - invested", positive=res["net_profit"] >= 0)
    if params["refi_enabled"] and res["cash_from_refi"] != 0:
        metric_card("Cash from Refi",  fmt_usd(res["cash_from_refi"]),
                    f"year {params['refi_year']}", positive=res["cash_from_refi"] > 0)
    else:
        metric_card("Monthly Mortgage", fmt_usd(res["monthly_payment"]), "P&I", positive=None)
    metric_card("Holding Period",  f"{params['holding_period']} yrs",
                f"exit {fmt_usd(params['purchase_price'] * (1+params['appr_rate']/100)**params['holding_period'])}", positive=None)

# ── charts ────────────────────────────────────────────────────────────────────
st.markdown("### Property Projections")

chart_col1, chart_col2 = st.columns(2)

# ── Property Value vs Loan Balance ───────────────────────────────────────────
with chart_col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["year"], y=df["prop_value"],
        name="Property Value", fill="tonexty",
        line=dict(color="#4f8ef7", width=2),
        fillcolor="rgba(79,142,247,0.15)"
    ))
    fig1.add_trace(go.Scatter(
        x=df["year"], y=df["loan_balance"],
        name="Loan Balance", fill="tozeroy",
        line=dict(color="#ff5252", width=2),
        fillcolor="rgba(255,82,82,0.15)"
    ))
    fig1.update_layout(
        title=dict(text="Property Value & Loan Balance", x=0.5, xanchor="center", pad=dict(b=8)),
        template="plotly_dark",
        xaxis_title="Year", yaxis_title="$",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=350, margin=dict(t=48, b=72, l=40, r=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Equity Over Time ──────────────────────────────────────────────────────────
with chart_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["year"], y=df["equity"],
        name="Equity", fill="tozeroy",
        line=dict(color="#00c853", width=2),
        fillcolor="rgba(0,200,83,0.2)"
    ))
    fig2.update_layout(
        title=dict(text="Equity Over Time", x=0.5, xanchor="center", pad=dict(b=8)),
        template="plotly_dark",
        xaxis_title="Year", yaxis_title="$",
        height=350, margin=dict(t=48, b=72, l=40, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

# ── Annual Cash Flow ──────────────────────────────────────────────────────────
with chart_col3:
    colors = ["#00c853" if v >= 0 else "#ff5252" for v in df["annual_cf"]]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df["year"], y=df["annual_cf"],
        marker_color=colors, name="Annual CF"
    ))
    fig3.add_trace(go.Scatter(
        x=df["year"], y=df["after_tax_cf"],
        name="After-Tax CF", line=dict(color="#ffd600", width=2, dash="dot")
    ))
    fig3.update_layout(
        title=dict(text="Annual Cash Flow Over Time", x=0.5, xanchor="center", pad=dict(b=8)),
        template="plotly_dark",
        xaxis_title="Year", yaxis_title="$",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=350, margin=dict(t=48, b=72, l=40, r=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Cumulative Profit ─────────────────────────────────────────────────────────
with chart_col4:
    cum_cf = np.cumsum(df["annual_cf"])
    cum_colors = ["#00c853" if v >= 0 else "#ff5252" for v in res["cumulative_profit"]]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df["year"], y=cum_cf,
        name="Cum. Cash Flow", line=dict(color="#4f8ef7", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=df["year"], y=res["cumulative_profit"],
        name="Cum. Profit (incl. equity)", fill="tozeroy",
        line=dict(color="#00c853", width=2),
        fillcolor="rgba(0,200,83,0.1)"
    ))
    fig4.update_layout(
        title=dict(text="Cumulative Profit Over Time", x=0.5, xanchor="center", pad=dict(b=8)),
        template="plotly_dark",
        xaxis_title="Year", yaxis_title="$",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=350, margin=dict(t=48, b=72, l=40, r=20)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Year-by-Year Table ────────────────────────────────────────────────────────
with st.expander("📋 Full Year-by-Year Projection"):
    display_df = df.copy()
    display_df.columns = [
        "Year", "Property Value", "Loan Balance", "Equity",
        "NOI", "Annual Cash Flow", "After-Tax CF",
        "Eff. Income", "Total OpEx", "Debt Service"
    ]
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Deal Comparison ───────────────────────────────────────────────────────────
if "compare_deals" in st.session_state and st.session_state["compare_deals"]:
    st.markdown("---")
    st.markdown("### 📊 Deal Comparison")
    deals = load_all_deals()
    compare_names = st.session_state["compare_deals"]

    rows = []
    chart_irr, chart_coc, chart_cf = [], [], []
    for name in compare_names:
        if name not in deals:
            continue
        r = run_model(deals[name])
        rows.append({
            "Deal": name,
            "Purchase Price": fmt_usd(deals[name]["purchase_price"]),
            "Down Payment":   fmt_usd(r["down"]),
            "Monthly CF":     fmt_usd(r["monthly_cf"]),
            "Annual CF":      fmt_usd(r["annual_cf"]),
            "NOI":            fmt_usd(r["noi"]),
            "Cap Rate":       fmt_pct(r["cap_rate"]),
            "CoC Return":     fmt_pct(r["coc"]),
            "IRR":            fmt_pct(r["irr"]),
            "After-Tax IRR":  fmt_pct(r["at_irr"]),
            "Equity at Exit": fmt_usd(r["equity_at_exit"]),
            "Net Profit":     fmt_usd(r["net_profit"]),
        })
        chart_irr.append(r["irr"])
        chart_coc.append(r["coc"])
        chart_cf.append(r["annual_cf"])

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="IRR (%)",         x=compare_names, y=chart_irr,  marker_color="#4f8ef7"))
        fig_cmp.add_trace(go.Bar(name="CoC Return (%)",  x=compare_names, y=chart_coc,  marker_color="#00c853"))
        fig_cmp.update_layout(
            barmode="group", template="plotly_dark",
            title="IRR vs Cash-on-Cash Return by Deal",
            height=350, margin=dict(t=40, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
