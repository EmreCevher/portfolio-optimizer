# ============================================================
# Turkish Stock Portfolio Simulator
# Interactive Decision-Support Dashboard
# ============================================================
# pip install streamlit yfinance pandas numpy matplotlib plotly PyPortfolioOpt scipy

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TICKERS   = ["THYAO.IS", "GARAN.IS", "ASELS.IS"]
TICKER_LABELS = {"THYAO.IS": "Turkish Airlines", "GARAN.IS": "Garanti Bank", "ASELS.IS": "Aselsan"}
BENCHMARK = "XU100.IS"
RISK_FREE_RATE = 0.40        # Annual risk-free rate (approx. Turkish 1-yr T-bill, adjust as needed)
PERIOD     = "1y"
NUM_RANDOM = 3000             # Random portfolios for scatter cloud

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portföy Simülatörü | Turkish Stocks",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background base */
.stApp {
    background: #0d0f14;
    color: #e8e8ec;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #1e2130;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161924 0%, #1a1e2e 100%);
    border: 1px solid #252a3d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4a6cf7; }
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7299;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #e8e8ec;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value.positive { color: #34d988; }
.metric-value.negative { color: #f05b5b; }
.metric-value.neutral  { color: #7b9cfc; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a6cf7;
    border-left: 3px solid #4a6cf7;
    padding-left: 12px;
    margin: 32px 0 16px 0;
}

/* Warning / info boxes */
.warn-box {
    background: rgba(240, 91, 91, 0.08);
    border: 1px solid rgba(240, 91, 91, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #f7a0a0;
}
.info-box {
    background: rgba(52, 217, 136, 0.06);
    border: 1px solid rgba(52, 217, 136, 0.25);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #80e8b8;
}
.neutral-box {
    background: rgba(74, 108, 247, 0.07);
    border: 1px solid rgba(74, 108, 247, 0.25);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #9ab3fc;
}

/* Asset table */
.asset-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #1e2130;
    font-size: 14px;
}

/* Divider */
hr { border-color: #1e2130 !important; }

/* Streamlit overrides */
.stSlider > div > div > div { background: #252a3d !important; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    """Download price data and compute returns for all assets + benchmark."""
    all_tickers = TICKERS + [BENCHMARK]
    raw = yf.download(all_tickers, period=PERIOD, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna()  # Keep overlapping dates only

    # Separate assets and benchmark
    asset_prices = raw[TICKERS]
    bench_prices = raw[BENCHMARK]

    # Daily returns
    asset_returns = asset_prices.pct_change().dropna()
    bench_returns  = bench_prices.pct_change().dropna()

    # Align
    common_idx = asset_returns.index.intersection(bench_returns.index)
    asset_returns = asset_returns.loc[common_idx]
    bench_returns  = bench_returns.loc[common_idx]

    return asset_returns, bench_returns


@st.cache_data(ttl=3600)
def compute_capm(asset_returns, bench_returns):
    """Compute betas and CAPM expected returns."""
    var_m = bench_returns.var() * 252          # Annualised market variance
    E_Rm  = bench_returns.mean() * 252          # Annualised market return

    betas = {}
    capm_returns = {}
    for ticker in TICKERS:
        cov_im = np.cov(asset_returns[ticker], bench_returns)[0, 1] * 252
        beta   = cov_im / var_m
        er     = RISK_FREE_RATE + beta * (E_Rm - RISK_FREE_RATE)
        betas[ticker]        = beta
        capm_returns[ticker] = er

    cov_matrix = asset_returns.cov() * 252     # Annualised covariance matrix
    return betas, capm_returns, cov_matrix, E_Rm


def portfolio_metrics(weights, capm_returns, cov_matrix):
    """Compute annualised return, volatility, and Sharpe ratio for given weights."""
    w = np.array(weights)
    ret = float(np.dot(w, [capm_returns[t] for t in TICKERS]))
    vol = float(np.sqrt(w @ cov_matrix.values @ w))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return ret, vol, sharpe


@st.cache_data(ttl=3600)
def compute_optimizer(_capm_returns, _cov_matrix):
    """Run Max-Sharpe optimisation via PyPortfolioOpt."""
    try:
        from pypfopt import EfficientFrontier, expected_returns as pf_er
        ef = EfficientFrontier(
            expected_returns=pd.Series(_capm_returns),
            cov_matrix=_cov_matrix,
            weight_bounds=(0, 1)
        )
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned = ef.clean_weights()
        opt_weights = [cleaned.get(t, 0) for t in TICKERS]
        opt_ret, opt_vol, opt_sharpe = portfolio_metrics(opt_weights, _capm_returns, _cov_matrix)
        return opt_weights, opt_ret, opt_vol, opt_sharpe, cleaned
    except Exception as e:
        return None, None, None, None, str(e)


@st.cache_data(ttl=3600)
def generate_random_portfolios(_capm_returns, _cov_matrix):
    """Simulate many random portfolios for the scatter cloud."""
    results = {"ret": [], "vol": [], "sharpe": []}
    for _ in range(NUM_RANDOM):
        w = np.random.dirichlet(np.ones(len(TICKERS)))
        r, v, s = portfolio_metrics(w, _capm_returns, _cov_matrix)
        results["ret"].append(r)
        results["vol"].append(v)
        results["sharpe"].append(s)
    return results


# ─────────────────────────────────────────────
# RISK & DIVERSIFICATION FUNCTIONS
# ─────────────────────────────────────────────
def compute_hhi(weights):
    w = np.array(weights)
    return float(np.sum(w ** 2))


def concentration_warnings(weights):
    warnings_list = []
    top2 = sorted(weights, reverse=True)[:2]
    for i, t in enumerate(TICKERS):
        if weights[i] > 0.30:
            warnings_list.append(f"⚠️  {t.replace('.IS','')} weight ({weights[i]:.0%}) exceeds 30% single-asset threshold.")
    if sum(top2) > 0.60:
        warnings_list.append(f"⚠️  Top-2 combined weight ({sum(top2):.0%}) exceeds 60% — portfolio is heavily concentrated.")
    return warnings_list


def interpret_portfolio(weights, vol, sharpe, hhi, opt_sharpe):
    """Generate natural-language interpretation of the user portfolio."""
    lines = []
    max_ticker = TICKERS[np.argmax(weights)]

    # Sharpe vs optimizer
    if opt_sharpe and sharpe < opt_sharpe * 0.8:
        lines.append(f"📉 Your Sharpe ratio is notably below the optimizer reference ({opt_sharpe:.2f}). Rebalancing may improve efficiency.")
    elif opt_sharpe and sharpe >= opt_sharpe * 0.95:
        lines.append("✅ Your portfolio efficiency is close to the optimizer reference — well done.")
    else:
        lines.append("📊 Your portfolio is moderately efficient relative to the optimizer reference.")

    # Concentration
    if max(weights) > 0.50:
        lines.append(f"🔴 Heavy concentration in {max_ticker.replace('.IS','')} — this adds idiosyncratic risk.")
    elif max(weights) > 0.30:
        lines.append(f"🟡 Moderate concentration in {max_ticker.replace('.IS','')} — consider spreading weight.")

    # HHI interpretation
    eff_n = 1 / hhi if hhi > 0 else len(TICKERS)
    if eff_n < 1.5:
        lines.append("🔴 Effective number of assets < 1.5 — portfolio behaves like a single-stock position.")
    elif eff_n < 2.2:
        lines.append(f"🟡 Effective number of assets ≈ {eff_n:.1f} — limited diversification benefit.")
    else:
        lines.append(f"🟢 Effective number of assets ≈ {eff_n:.1f} — reasonable diversification.")

    return lines


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def build_chart(random_data, user_ret, user_vol, opt_ret, opt_vol, user_sharpe, opt_sharpe):
    fig = go.Figure()

    # Random portfolio scatter (coloured by Sharpe)
    fig.add_trace(go.Scatter(
        x=random_data["vol"],
        y=random_data["ret"],
        mode="markers",
        marker=dict(
            size=4,
            color=random_data["sharpe"],
            colorscale="Viridis",
            opacity=0.55,
            showscale=True,
            colorbar=dict(title="Sharpe", thickness=12, len=0.6, tickfont=dict(color="#8898b8"))
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Optimizer portfolio
    if opt_ret is not None:
        fig.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret],
            mode="markers+text",
            marker=dict(size=18, color="#f5a623", symbol="star", line=dict(color="#fff", width=1.5)),
            text=["Optimizer"],
            textposition="top right",
            textfont=dict(color="#f5a623", size=12, family="IBM Plex Mono"),
            name=f"Optimizer (Sharpe {opt_sharpe:.2f})",
            hovertemplate=f"Optimizer<br>Vol: {opt_vol:.2%}<br>Ret: {opt_ret:.2%}<br>Sharpe: {opt_sharpe:.2f}<extra></extra>",
        ))

    # User portfolio
    fig.add_trace(go.Scatter(
        x=[user_vol], y=[user_ret],
        mode="markers+text",
        marker=dict(size=18, color="#4a6cf7", symbol="diamond", line=dict(color="#fff", width=1.5)),
        text=["Your Portfolio"],
        textposition="top left",
        textfont=dict(color="#7b9cfc", size=12, family="IBM Plex Mono"),
        name=f"Your Portfolio (Sharpe {user_sharpe:.2f})",
        hovertemplate=f"Your Portfolio<br>Vol: {user_vol:.2%}<br>Ret: {user_ret:.2%}<br>Sharpe: {user_sharpe:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Risk–Return Space",
            font=dict(family="IBM Plex Mono", size=14, color="#8898b8"),
            x=0.01,
        ),
        xaxis=dict(
            title="Annualised Volatility",
            tickformat=".0%",
            gridcolor="#1a1e2e",
            linecolor="#252a3d",
            color="#8898b8",
        ),
        yaxis=dict(
            title="Annualised Expected Return (CAPM)",
            tickformat=".0%",
            gridcolor="#1a1e2e",
            linecolor="#252a3d",
            color="#8898b8",
        ),
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        legend=dict(
            bgcolor="#13161f",
            bordercolor="#252a3d",
            borderwidth=1,
            font=dict(color="#8898b8", size=11),
        ),
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────
    st.markdown("""
    <div style="padding: 8px 0 32px 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:3px;
                    text-transform:uppercase; color:#4a6cf7; margin-bottom:8px;">
            TÜRK HİSSE SENETLERİ
        </div>
        <h1 style="font-family:'DM Sans',sans-serif; font-size:36px; font-weight:700;
                   color:#e8e8ec; margin:0 0 10px 0; line-height:1.1;">
            Portfolio Simulator
        </h1>
        <p style="color:#6b7299; font-size:15px; max-width:600px; margin:0;">
            Adjust asset weights using the sliders on the left. The dashboard instantly
            updates expected return, risk, and Sharpe ratio — compare your choices against
            the mean-variance optimizer reference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────
    with st.spinner("Fetching market data from Yahoo Finance…"):
        try:
            asset_returns, bench_returns = load_data()
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    betas, capm_returns, cov_matrix, E_Rm = compute_capm(asset_returns, bench_returns)

    # ── Sidebar / Weight Controls ─────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:2px;
                    color:#4a6cf7; text-transform:uppercase; margin-bottom:20px; padding-top:8px;">
            ⚙ Portfolio Weights
        </div>
        """, unsafe_allow_html=True)

        raw_weights = {}
        for t in TICKERS:
            label = f"{t.replace('.IS','')}  ·  {TICKER_LABELS[t]}"
            raw_weights[t] = st.slider(label, 0, 100, 33, step=1, key=t)

        total_raw = sum(raw_weights.values())
        st.markdown("---")

        if total_raw == 0:
            st.warning("All weights are zero. Using equal weights.")
            weights = {t: 1/len(TICKERS) for t in TICKERS}
        else:
            weights = {t: raw_weights[t] / total_raw for t in TICKERS}

        st.markdown("""<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                    letter-spacing:2px; color:#4a6cf7; text-transform:uppercase;
                    margin-bottom:10px;">Normalised Weights</div>""", unsafe_allow_html=True)
        for t in TICKERS:
            bar_w = int(weights[t] * 100)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#c8cce8; font-size:13px;">{t.replace('.IS','')}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; color:#7b9cfc; font-size:13px;">
                        {weights[t]:.1%}
                    </span>
                </div>
                <div style="background:#1e2130; border-radius:4px; height:5px;">
                    <div style="background:#4a6cf7; width:{bar_w}%; height:5px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:12px; color:#6b7299; line-height:1.7;">
            <b style="color:#8898b8;">Risk-Free Rate</b><br>{RISK_FREE_RATE:.0%} annual (Turkish T-bill)<br><br>
            <b style="color:#8898b8;">Market Benchmark</b><br>BIST 100 (XU100.IS)<br><br>
            <b style="color:#8898b8;">Data</b><br>Last 1 year, daily prices<br><br>
            <b style="color:#8898b8;">Expected Returns</b><br>CAPM (not historical mean)
        </div>
        """, unsafe_allow_html=True)

    # ── Compute User Portfolio ─────────────────
    weight_list = [weights[t] for t in TICKERS]
    user_ret, user_vol, user_sharpe = portfolio_metrics(weight_list, capm_returns, cov_matrix)
    hhi = compute_hhi(weight_list)
    eff_n = 1 / hhi if hhi > 0 else len(TICKERS)

    # ── Optimizer ─────────────────────────────
    opt_weights, opt_ret, opt_vol, opt_sharpe, opt_detail = compute_optimizer(
        capm_returns, cov_matrix
    )

    # ── Main Metrics ──────────────────────────
    st.markdown('<div class="section-header">Portfolio Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    def metric_html(label, value, cls="neutral"):
        return f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>"""

    ret_cls  = "positive" if user_ret > 0 else "negative"
    shp_cls  = "positive" if user_sharpe > 1 else ("neutral" if user_sharpe > 0 else "negative")

    with c1:
        st.markdown(metric_html("Expected Return", f"{user_ret:.2%}", ret_cls), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_html("Volatility", f"{user_vol:.2%}", "neutral"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_html("Sharpe Ratio", f"{user_sharpe:.2f}", shp_cls), unsafe_allow_html=True)
    with c4:
        total_display = sum(raw_weights.values())
        tw_cls = "positive" if total_display > 0 else "negative"
        st.markdown(metric_html("Raw Weight Sum", f"{total_display}%", tw_cls), unsafe_allow_html=True)

    # ── CAPM Asset Diagnostics ────────────────
    st.markdown('<div class="section-header">CAPM Asset Diagnostics</div>', unsafe_allow_html=True)

    diag_cols = st.columns(len(TICKERS))
    for i, t in enumerate(TICKERS):
        with diag_cols[i]:
            beta_val = betas[t]
            er_val   = capm_returns[t]
            beta_cls = "positive" if beta_val < 1 else ("neutral" if beta_val < 1.5 else "negative")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{t.replace('.IS','')} · {TICKER_LABELS[t]}</div>
                <div style="margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="color:#6b7299; font-size:12px; font-family:'IBM Plex Mono',monospace;">BETA</span>
                        <span style="color:#e8e8ec; font-size:14px; font-family:'IBM Plex Mono',monospace;">{beta_val:.3f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#6b7299; font-size:12px; font-family:'IBM Plex Mono',monospace;">CAPM E(R)</span>
                        <span style="color:#34d988; font-size:14px; font-family:'IBM Plex Mono',monospace;">{er_val:.2%}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:12px; color:#6b7299; margin-top:8px;">
        Market E(Rₘ) = <span style="color:#c8cce8;">{E_Rm:.2%}</span> &nbsp;·&nbsp;
        Risk-Free Rate = <span style="color:#c8cce8;">{RISK_FREE_RATE:.2%}</span> &nbsp;·&nbsp;
        Market Premium = <span style="color:#c8cce8;">{(E_Rm - RISK_FREE_RATE):.2%}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────
    st.markdown('<div class="section-header">Risk–Return Chart</div>', unsafe_allow_html=True)
    with st.spinner("Generating portfolio cloud…"):
        random_data = generate_random_portfolios(capm_returns, cov_matrix)

    fig = build_chart(random_data, user_ret, user_vol, opt_ret, opt_vol, user_sharpe, opt_sharpe)
    st.plotly_chart(fig, use_container_width=True)

    # ── Optimizer Reference ───────────────────
    st.markdown('<div class="section-header">Optimizer Reference (Max-Sharpe)</div>', unsafe_allow_html=True)

    if opt_weights is not None:
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown(metric_html("Opt. Return", f"{opt_ret:.2%}", "positive"), unsafe_allow_html=True)
        with oc2:
            st.markdown(metric_html("Opt. Volatility", f"{opt_vol:.2%}", "neutral"), unsafe_allow_html=True)
        with oc3:
            st.markdown(metric_html("Opt. Sharpe", f"{opt_sharpe:.2f}", "positive"), unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        opt_row = " &nbsp;|&nbsp; ".join(
            [f"<span style='color:#8898b8;'>{t.replace('.IS','')}</span> "
             f"<span style='color:#f5a623; font-family:IBM Plex Mono,monospace;'>{v:.1%}</span>"
             for t, v in zip(TICKERS, opt_weights)]
        )
        st.markdown(f"""
        <div class="neutral-box">
            <strong>Optimizer Weights →</strong> {opt_row}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Optimiser error: {opt_detail}")

    # ── Risk & Diversification ────────────────
    st.markdown('<div class="section-header">Risk & Diversification Analysis</div>', unsafe_allow_html=True)

    warns = concentration_warnings(weight_list)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Concentration Warnings**")
        if warns:
            for w in warns:
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ No concentration warnings — weights are well-distributed.</div>',
                        unsafe_allow_html=True)

    with col_right:
        st.markdown("**Diversification Metrics**")
        has_concentration = len(warns) > 0

        if eff_n >= 2.5 and not has_concentration:
            div_label = "🟢 Excellent"
            div_cls   = "info-box"
        elif eff_n >= 1.8 or has_concentration:
            div_label = "🟡 Moderate"
            div_cls   = "warn-box"
        else:
            div_label = "🔴 Poor"
            div_cls   = "warn-box"

        st.markdown(f"""
        <div class="{div_cls}">
            <div style="margin-bottom:6px;"><strong>HHI (Herfindahl–Hirschman Index)</strong><br>
            <span style="font-family:'IBM Plex Mono',monospace;">{hhi:.4f}</span>
            <span style="font-size:12px; margin-left:8px;">(1.0 = monopoly, {1/len(TICKERS):.4f} = perfect equal split)</span></div>
            <div style="margin-bottom:6px;"><strong>Effective # of Assets</strong><br>
            <span style="font-family:'IBM Plex Mono',monospace;">{eff_n:.2f}</span> / {len(TICKERS):.0f}</div>
            <div><strong>Diversification Quality</strong><br>{div_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Separate risk dimensions
    st.markdown("**Risk Dimensions**")
    vol_label  = "🟢 Low"   if user_vol < 0.20 else ("🟡 Medium" if user_vol < 0.35 else "🔴 High")
    conc_label = "🔴 High"  if len(warns) >= 2 else ("🟡 Medium" if len(warns) == 1 else "🟢 Low")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f'<div class="neutral-box"><strong>Volatility Risk</strong><br>{vol_label}<br>'
                    f'<span style="font-family:IBM Plex Mono,monospace;">{user_vol:.2%} annual</span></div>',
                    unsafe_allow_html=True)
    with rc2:
        st.markdown(f'<div class="neutral-box"><strong>Concentration Risk</strong><br>{conc_label}<br>'
                    f'<span style="font-size:12px;">{len(warns)} active warning(s)</span></div>',
                    unsafe_allow_html=True)
    with rc3:
        div_qual = div_label
        st.markdown(f'<div class="neutral-box"><strong>Diversification Quality</strong><br>{div_qual}<br>'
                    f'<span style="font-size:12px;">HHI = {hhi:.4f}</span></div>',
                    unsafe_allow_html=True)

    # ── Interpretation ────────────────────────
    st.markdown('<div class="section-header">Portfolio Interpretation</div>', unsafe_allow_html=True)
    interp = interpret_portfolio(weight_list, user_vol, user_sharpe, hhi, opt_sharpe)
    for line in interp:
        cls = "info-box" if line.startswith("✅") or line.startswith("🟢") else \
              ("warn-box" if line.startswith("🔴") or line.startswith("📉") else "neutral-box")
        st.markdown(f'<div class="{cls}">{line}</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#3a3f5c; padding: 12px 0 4px 0;
                font-family:'IBM Plex Mono',monospace; letter-spacing:1px;">
        TURKISH PORTFOLIO SIMULATOR · CAPM-BASED · FOR EDUCATIONAL USE ONLY<br>
        Data: Yahoo Finance · Optimizer: PyPortfolioOpt · Not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
