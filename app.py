# pyrefly: ignore-errors
import warnings
import yaml

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import our custom modules
import models
from data_loader import get_data_splits

warnings.filterwarnings("ignore")

# Must be the first Streamlit command
st.set_page_config(
    page_title="CS2 Time Series",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CS2 Tactical Palette
CS2_CT_BLUE = "#5D79AE"
CS2_T_GOLD = "#E5A823"
CS2_GUNMETAL = "#121418"
CS2_WARNING = "#ff7b72"


# ── HACKING STREAMLIT: SaaS Mode CSS ──
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* HIDE STREAMLIT CHROME (Header, Menu, Footer) */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Remove top padding to make it look like a native app */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #121418 !important;
    border-right: 1px solid #1f2329;
}
.nav-section {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5D79AE; /* CT Blue */
    margin: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #1f2329;
    padding-bottom: 0.3rem;
}

/* Stat Cards (Tactical Glass) */
.stat-card {
    background: rgba(22, 27, 34, 0.7);
    backdrop-filter: blur(10px);
    padding: 1.2rem;
    border-radius: 4px;
    border: 1px solid #1f2329;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: #E5A823; /* T Gold */
}
.stat-lbl {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.stat-val {
    font-size: 2.5rem;
    font-weight: 700;
    color: #E5A823; /* T Gold */
    line-height: 1;
    text-shadow: 0 0 10px rgba(229, 168, 35, 0.3);
}
.stat-note {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.4rem;
}

/* Findings / Insights Boxes */
.finding {
    background: rgba(28, 33, 40, 0.8);
    backdrop-filter: blur(10px);
    border-left: 4px solid #5D79AE; /* CT Blue */
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
    font-size: 1.1rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.finding b { color: #ffffff; font-weight: 700; }
.finding.key { border-left-color: #E5A823; background: rgba(42, 25, 29, 0.8); }

/* Analysis Text */
.analysis {
    font-size: 1.1rem;
    color: #8b949e;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── DATA LOADING ──
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {
        "models": {
            "arima": {"order": [2, 1, 2], "steps": 12},
            "var": {"maxlags": 12, "steps": 12},
            "garch": {"p": 1, "q": 1},
            "cross_correlation": {"max_lags": 12},
            "acf_pacf": {"nlags": 20},
        }
    }

df, df_full, df_multi = get_data_splits()

# Plotly Theme Settings
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#c9d1d9"

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("###  CS2 Analytics")
    st.markdown('<div class="nav-section">Dashboard View</div>', unsafe_allow_html=True)
    section = st.radio(
        "",
        [
            "1. EDA Overview",
            "2. Stationarity",
            "3. ARIMA Forecast",
            "4. VAR + Granger",
            "5. GARCH Volatility",
            "6. Summary",
            "7. Advanced Diagnostics",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="nav-section" style="margin-top:2rem">Data Specs</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"**Total Records:** {len(df)}")
    st.caption("**Time Horizon:** Jul 2012 — Apr 2026")
    st.caption(f"**Multivariate Overlap:** {len(df_multi)} months")

# ── PAGE HEADER ──
st.title("Counter-Strike 2 Engagement Ecosystem")
st.markdown("*Multivariate Time Series Analysis · Steam & Twitch · 2012–2026*")
st.divider()

# ══════════════════════════════════════════
# 1. EDA OVERVIEW
# ══════════════════════════════════════════
if section == "1. EDA Overview":
    st.subheader("Exploratory Data Analysis")
    st.markdown(
        '<div class="analysis">Analyzing the three core dimensions of game '
        "engagement: Active participation (Steam), Public awareness (Google), "
        "and Spectatorship (Twitch).</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Players</div>'
        f'<div class="stat-val">{df["Players"].max()/1e6:.2f}M</div>'
        '<div class="stat-note">May 2023</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Trends Index</div>'
        f'<div class="stat-val">{int(df["Trends"].max())}</div>'
        '<div class="stat-note">Aug 2012</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Twitch Viewers'
        f'</div><div class="stat-val">{int(df_multi["Twitch"].max()/1e3)}K'
        '</div><div class="stat-note">Concurrent Avg</div></div>',
        unsafe_allow_html=True,
    )

    # PLOTLY INTERACTIVE CHART
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Steam — Monthly Active Players",
            "Google Trends — Search Index",
            "Twitch — Concurrent Viewers",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full["Players"],
            fill="tozeroy",
            name="Players",
            line=dict(color=CS2_CT_BLUE, width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full["Trends"],
            fill="tozeroy",
            name="Trends",
            line=dict(color=CS2_T_GOLD, width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_multi.index,
            y=df_multi["Twitch"],
            fill="tozeroy",
            name="Twitch",
            line=dict(color="#8b949e", width=2),
        ),
        row=3,
        col=1,
    )

    # Add CS2 release markers
    for i in range(1, 4):
        fig.add_vline(
            x=pd.to_datetime("2023-09-01").timestamp() * 1000,
            line_width=1,
            line_dash="dash",
            line_color=CS2_WARNING,
            row=i,
            col=1,
        )

    fig.update_layout(
        height=800,
        showlegend=False,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>The Core Ecosystem:</b> This dashboard tracks three interconnected data streams. Player Count (Blue) is our ground truth. We can immediately observe a <b>Divergence of Trends and Players</b>—Google search interest (Gold) has slowly declined over the decade, while actual active players exploded. This proves that CS:GO/CS2 relies on a loyal, habitual user base rather than constant viral hype.</div>'
        '<div class="finding"><b>Structural Break:</b> The massive spike and subsequent stabilization in late 2023 clearly marks the CS2 release, which established a completely new baseline equilibrium for the franchise.</div>'
        '<div class="finding"><b>Twitch Data Limitations:</b> You may notice that the Twitch viewer data (Grey) only begins in December 2016. Because Twitch was still emerging as a mainstream platform in the early years of CS:GO, historical viewership data prior to late 2016 is unavailable and naturally appears as empty rows in the dataset.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Seasonal Decomposition (Players)")
    trend, seasonal, resid = models.get_seasonal_decomposition(df_full["Players"])
    fig_sd = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Trend", "Seasonal", "Residual"),
    )
    fig_sd.add_trace(
        go.Scatter(x=trend.index, y=trend, line=dict(color=CS2_CT_BLUE)), row=1, col=1
    )
    fig_sd.add_trace(
        go.Scatter(x=seasonal.index, y=seasonal, line=dict(color=CS2_T_GOLD)),
        row=2,
        col=1,
    )
    fig_sd.add_trace(
        go.Scatter(x=resid.index, y=resid, line=dict(color=CS2_WARNING)), row=3, col=1
    )
    fig_sd.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig_sd.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig_sd.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_sd, use_container_width=True)

    st.markdown(
        '<div class="finding"><b>Deconstructing the Time Series:</b> By mathematically stripping away the underlying trajectory (Trend), we isolate the pure <b>Seasonal</b> heartbeat of the game. We can clearly observe recurring annual spikes during winter holidays and summer breaks. The <b>Residual</b> plot highlights pure noise or unexpected shocks (like the massive influx during the COVID-19 lockdowns) after accounting for the trend and seasonality.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
# 2. STATIONARITY
# ══════════════════════════════════════════
elif section == "2. Stationarity":
    st.subheader("Augmented Dickey-Fuller Test")
    st.markdown(
        '<div class="analysis">Evaluating unit roots before modelling. '
        "Tests are conducted in levels and after first differencing "
        "(d=1).</div>",
        unsafe_allow_html=True,
    )

    def format_adf_row(name, series):
        stat, pval, is_stat = models.get_adf_test_results(series)
        return (name, stat, pval)

    rows_orig = [
        format_adf_row("Players", df_full["Players"]),
        format_adf_row("Trends", df_full["Trends"]),
        format_adf_row("Twitch", df_multi["Twitch"]),
    ]
    rows_diff = [
        format_adf_row("Players (d=1)", df_full["Players"].diff()),
        format_adf_row("Trends (d=1)", df_full["Trends"].diff()),
        format_adf_row("Twitch (d=1)", df_multi["Twitch"].diff()),
    ]

    def render_adf(rows):
        df_adf = pd.DataFrame(rows, columns=["Variable", "ADF Statistic", "p-value"])
        df_adf["Result"] = df_adf["p-value"].apply(
            lambda x: "Stationary" if x < 0.05 else "Non-Stationary"
        )

        def style_fn(v):
            if v == "Stationary":
                return "color: #3fb950; font-weight: bold;"
            return "color: #ff7b72; font-weight: bold;"

        st.dataframe(
            df_adf.style.map(style_fn, subset=["Result"]),
            hide_index=True,
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Level Series**")
        render_adf(rows_orig)
    with c2:
        st.markdown("**First Difference (d=1)**")
        render_adf(rows_diff)

    # PLOTLY INTERACTIVE CHART
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Δ Players", "Δ Trends", "Δ Twitch"),
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full["Players"].diff(),
            line=dict(color=CS2_CT_BLUE, width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full["Trends"].diff(),
            line=dict(color=CS2_T_GOLD, width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_multi.index,
            y=df_multi["Twitch"].diff(),
            line=dict(color="#8b949e", width=1.5),
        ),
        row=3,
        col=1,
    )

    for i in range(1, 4):
        fig.add_hline(
            y=0, line_width=1, line_color="#8b949e", opacity=0.5, row=i, col=1
        )

    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding"><b>Why take the First Difference?</b> Predictive models like ARIMA mathematically break if data wanders infinitely upwards (non-stationary). The ADF test proves our raw data is non-stationary (Red). By taking the "First Difference" (measuring month-to-month change instead of absolute totals), we force the data to center around zero, making it mathematically stable (Green) and ready for forecasting.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### ACF & PACF (Differenced Players)")
    acf_vals, acf_conf, pacf_vals, pacf_conf = models.get_acf_pacf_values(
        df_full["Players"].diff(), nlags=config["models"]["acf_pacf"]["nlags"]
    )
    fig_acf = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"),
    )

    # ACF
    fig_acf.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color=CS2_CT_BLUE),
        row=1,
        col=1,
    )
    fig_acf.add_trace(
        go.Scatter(
            x=list(range(len(acf_vals))),
            y=acf_conf[:, 0] - acf_vals,
            mode="lines",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_acf.add_trace(
        go.Scatter(
            x=list(range(len(acf_vals))),
            y=acf_conf[:, 1] - acf_vals,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,255,255,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # PACF
    fig_acf.add_trace(
        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color=CS2_T_GOLD),
        row=1,
        col=2,
    )
    fig_acf.add_trace(
        go.Scatter(
            x=list(range(len(pacf_vals))),
            y=pacf_conf[:, 0] - pacf_vals,
            mode="lines",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_acf.add_trace(
        go.Scatter(
            x=list(range(len(pacf_vals))),
            y=pacf_conf[:, 1] - pacf_vals,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,255,255,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_acf.update_layout(
        height=350,
        showlegend=False,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig_acf.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig_acf.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_acf, use_container_width=True)

# ══════════════════════════════════════════
# 3. ARIMA FORECAST
# ══════════════════════════════════════════
elif section == "3. ARIMA Forecast":
    arima_order = tuple(config["models"]["arima"]["order"])
    st.subheader(f"ARIMA{arima_order} — Player Count Projections")

    st.markdown(
        '<div class="analysis">Use the sliders below to dynamically adjust the forecasting horizon and confidence interval risk parameters. The model will optimize and re-render the cone of uncertainty in real-time.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        arima_steps = st.slider(
            "Forecast Horizon (Months)",
            min_value=3,
            max_value=24,
            value=config["models"]["arima"]["steps"],
        )
    with col_b:
        ci_level = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1)

    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    ci_alpha = alpha_map[ci_level]

    with st.spinner("Optimizing ARIMA model..."):
        players = df_full["Players"]
        fitted, fc_mean, fc_ci = models.get_arima_forecast(
            players, order=arima_order, steps=arima_steps, alpha=ci_alpha
        )

        # Calculate in-sample metrics
        in_sample_pred = fitted.predict(start=1, end=len(players) - 1)
        metrics = models.compute_forecast_metrics(players.iloc[1:], in_sample_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Model Architecture</div>'
        f'<div class="stat-val">ARIMA{arima_order}</div>'
        '<div class="stat-note">Selected via Min AIC</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">AIC Score</div>'
        f'<div class="stat-val">{fitted.aic:.1f}</div>'
        '<div class="stat-note">Goodness of fit</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">In-Sample MAPE</div>'
        f'<div class="stat-val">{metrics["MAPE"]:.2f}%</div>'
        '<div class="stat-note">Mean Abs Pct Error</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        '<div class="stat-card"><div class="stat-lbl">12-Mo Trajectory</div>'
        f'<div class="stat-val">{fc_mean.mean()/1e3:.0f}K</div>'
        '<div class="stat-note">Projected Baseline</div></div>',
        unsafe_allow_html=True,
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=players.index,
            y=players,
            name="Actual Players",
            line=dict(color=CS2_CT_BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_mean.index,
            y=fc_mean,
            name="Forecast",
            line=dict(color=CS2_WARNING, width=3, dash="dash"),
        )
    )

    # Confidence Interval Shading
    concat_x = pd.concat([pd.Series(fc_ci.index), pd.Series(fc_ci.index[::-1])])
    concat_y = pd.concat([fc_ci.iloc[:, 0], fc_ci.iloc[::-1, 1]])

    fig.add_trace(
        go.Scatter(
            x=concat_x,
            y=concat_y,
            fill="toself",
            fillcolor="rgba(255, 123, 114, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=450,
        title="Steam Player Count Forecast (12 Months)",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Interactive Forecast Data"):
        fc_table = pd.DataFrame(
            {
                "Predicted Players": fc_mean.round(0),
                "Lower Bound": fc_ci.iloc[:, 0].round(0),
                "Upper Bound": fc_ci.iloc[:, 1].round(0),
            }
        )
        st.dataframe(fc_table, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>Interpreting the Forecast (ARIMA):</b> ARIMA (Auto-Regressive Integrated Moving Average) is a univariate model, meaning it relies entirely on the historical momentum of a single variable (Player Count) and ignores all outside factors like esports tournaments or game updates. Because the CS2 release triggered a massive, prolonged plateau in player count, the algorithm mathematically projects this flat momentum into the future. It establishes a stabilization baseline right around 960K players. This strongly implies that without a massive external catalyst (like a new Operation or major content drop), organic player growth has temporarily capped and stabilized.</div>'
        '<div class="finding"><b>The Cone of Uncertainty & Risk:</b> The shaded red area represents the 95% confidence interval. Notice how the cone drastically widens over time. By Month 12, the model mathematically acknowledges that the player base could surge past 1.2M or drop below 700K. This mathematical margin of error proves that long-term univariate forecasting is highly risky in gaming, where sudden patches or competing game releases can instantly break historical patterns.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
# 4. VAR + GRANGER
# ══════════════════════════════════════════
elif section == "4. VAR + Granger":
    st.subheader("Multivariate Dynamics (VAR) & Causality")

    var_maxlags = config["models"]["var"]["maxlags"]

    st.markdown(
        '<div class="analysis">Adjust the VAR multivariate projection timeframe. Notice how the model predicts organic waves rather than a flat line.</div>',
        unsafe_allow_html=True,
    )
    var_steps = st.slider(
        "VAR Forecast Horizon (Months)",
        min_value=3,
        max_value=24,
        value=config["models"]["var"]["steps"],
        key="var_slider",
    )

    var_data = df_multi[["Players", "Trends", "Twitch"]].diff().dropna()
    with st.spinner("Fitting VAR model..."):
        best_lag, vf, fc_df2 = models.get_var_forecast(
            var_data, maxlags=var_maxlags, steps=var_steps
        )

    afc = [df_multi["Players"].iloc[-1]]
    for d in fc_df2["Players"]:
        afc.append(afc[-1] + d)

    fc_dates = pd.date_range(start=df_multi.index[-1], periods=var_steps + 1, freq="MS")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Optimal Lag</div>'
        f'<div class="stat-val">L={best_lag}</div>'
        '<div class="stat-note">Months of look-back</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">System</div>'
        '<div class="stat-val">3-Var</div>'
        '<div class="stat-note">Players / Trends / Twitch</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">Data Window</div>'
        '<div class="stat-val">2017+</div>'
        '<div class="stat-note">Multivariate intersection</div></div>',
        unsafe_allow_html=True,
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_multi.index,
            y=df_multi["Players"],
            name="Actual",
            line=dict(color=CS2_CT_BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=afc,
            name="VAR Forecast",
            line=dict(color=CS2_WARNING, width=3, dash="dash"),
        )
    )
    fig.update_layout(
        height=450,
        title="System-Aware Player Forecast (VAR)",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Interactive VAR Predictions"):
        var_table = pd.DataFrame(
            {"Date": fc_dates, "Predicted Players": [round(x, 0) for x in afc]}
        ).set_index("Date")
        st.dataframe(var_table, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>System-Aware Forecasting (VAR):</b> Unlike ARIMA, Vector Autoregression (VAR) understands that the gaming ecosystem is interconnected. By analyzing Players, Trends, and Twitch simultaneously across an optimal 5-month lag window, the model anticipates nuanced cyclical movements. Notice how the VAR forecast curve is not a flat line like ARIMA—it predicts organic dips and recoveries because it factors in the delayed impact of search interest (Trends) bleeding into actual gameplay.</div>'
        '<div class="finding"><b>Prediction Details:</b> The VAR model specifically predicts a slight near-term contraction in the player base followed by a seasonal rebound, driven by the historical cycles of Twitch viewership and Google search traffic leading up to major tournaments. This makes VAR vastly superior for short-term strategic planning, as it "sees" the incoming momentum from social and search channels before it hits the Steam servers.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Granger Causality Inference")
    c1, c2 = st.columns(2)

    def style_fn(v):
        if v == "Significant":
            return "color: #3fb950; font-weight: bold;"
        return "color: #8b949e;"

    with c1:
        st.markdown("**Does Search Volume (Trends) predict Players?**")
        df_trends = models.get_granger_causality(
            var_data[["Players", "Trends"]], maxlag=best_lag
        )
        st.dataframe(
            df_trends.style.map(style_fn, subset=["Signal"]),
            hide_index=True,
            use_container_width=True,
        )

    with c2:
        st.markdown("**Do Viewers (Twitch) predict Players?**")
        df_twitch = models.get_granger_causality(
            var_data[["Players", "Twitch"]], maxlag=best_lag
        )
        st.dataframe(
            df_twitch.style.map(style_fn, subset=["Signal"]),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Cross-Correlation")
    ccf_max_lags = config["models"]["cross_correlation"]["max_lags"]
    df_ccf = models.get_cross_correlation(
        var_data["Trends"], var_data["Players"], max_lags=ccf_max_lags
    )
    fig_ccf = go.Figure(
        go.Bar(x=df_ccf["Lag"], y=df_ccf["Correlation"], marker_color=CS2_CT_BLUE)
    )
    fig_ccf.update_layout(
        height=300,
        title="Cross-Correlation: Trends vs Players",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    st.plotly_chart(fig_ccf, use_container_width=True)

    st.markdown(
        '<div class="finding"><b>Visualizing the Lead:</b> This Cross-Correlation (CCF) bar chart visualizes exactly <i>when</i> the strongest relationships occur. We observe the highest correlation spikes happening at specific monthly lags, visually confirming the Granger causality results that search interest precedes gameplay activity.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
# 5. GARCH VOLATILITY
# ══════════════════════════════════════════
elif section == "5. GARCH Volatility":
    garch_p = config["models"]["garch"]["p"]
    garch_q = config["models"]["garch"]["q"]
    st.subheader(f"GARCH({garch_p},{garch_q}) — Volatility Clustering")

    with st.spinner("Estimating Conditional Variance..."):
        pd_ = df_full["Players"].diff().dropna()
        vol, prm = models.get_garch_volatility(pd_, p=garch_p, q=garch_q)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Alpha (α)</div>'
        f'<div class="stat-val">{prm["alpha[1]"]:.4f}</div>'
        '<div class="stat-note">Shock Reactivity</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">Beta (β)</div>'
        f'<div class="stat-val">{prm["beta[1]"]:.4f}</div>'
        '<div class="stat-note">Volatility Persistence</div></div>',
        unsafe_allow_html=True,
    )

    mem_factor = prm["alpha[1]"] + prm["beta[1]"]
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">α + β</div>'
        f'<div class="stat-val">{mem_factor:.4f}</div>'
        '<div class="stat-note">Memory Factor (~1.0 is high)</div></div>',
        unsafe_allow_html=True,
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol.index,
            y=vol,
            fill="tozeroy",
            name="Volatility",
            line=dict(color=CS2_WARNING, width=2),
        )
    )
    fig.add_vline(
        x=pd.to_datetime("2020-03-01").timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="#8b949e",
        annotation_text="COVID-19",
    )
    fig.add_vline(
        x=pd.to_datetime("2023-09-01").timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="#8b949e",
        annotation_text="CS2 Release",
    )
    fig.update_layout(
        height=450,
        title="Conditional Volatility of Player Base",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>Modeling Volatility Clusters (GARCH):</b> Financial models like GARCH don\'t predict <i>how many</i> players there will be; they predict <i>how chaotic</i> the player count will be. Alpha (α) measures sensitivity to new shocks, while Beta (β) measures how long the chaos lasts.</div>'
        '<div class="finding"><b>Long Memory Dynamics:</b> With α + β approaching 1.0, the model proves that Counter-Strike experiences "long memory" volatility. Major shocks—like global pandemics or the release of CS2—create mathematical ripples of uncertainty that take over a year to fully stabilize.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
# 7. ADVANCED DIAGNOSTICS
# ══════════════════════════════════════════
elif section == "7. Advanced Diagnostics":
    st.subheader("Rolling Statistics, Anomalies & Growth Momentum")

    c1, c2, c3 = st.columns(3)

    mom, yoy = models.get_growth_rates(df_full["Players"])
    latest_mom = mom.iloc[-1]
    latest_yoy = yoy.iloc[-1]

    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">MoM Growth</div>'
        f'<div class="stat-val" style="color: {CS2_CT_BLUE if latest_mom > 0 else CS2_WARNING}">{latest_mom:+.1f}%</div>'
        '<div class="stat-note">Month-over-Month</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">YoY Growth</div>'
        f'<div class="stat-val" style="color: {CS2_CT_BLUE if latest_yoy > 0 else CS2_WARNING}">{latest_yoy:+.1f}%</div>'
        '<div class="stat-note">Year-over-Year</div></div>',
        unsafe_allow_html=True,
    )

    anomalies = models.get_anomalies(df_full["Players"])
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">Total Anomalies</div>'
        f'<div class="stat-val" style="color: {CS2_T_GOLD}">{len(anomalies)}</div>'
        '<div class="stat-note">Spikes > 2.5σ</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 12-Month Rolling Baseline")
    rmean, rstd = models.get_rolling_statistics(df_full["Players"], window=12)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_full.index,
            y=df_full["Players"],
            name="Raw Players",
            line=dict(color=f"rgba(93, 121, 174, 0.3)", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rmean.index,
            y=rmean,
            name="12M Rolling Mean",
            line=dict(color=CS2_T_GOLD, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies.index,
            y=anomalies.values,
            mode="markers",
            name="Anomalies",
            marker=dict(color=CS2_WARNING, size=8, symbol="x"),
        )
    )

    fig.update_layout(
        height=450,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding"><b>Macro Baseline Growth:</b> The 12-month rolling mean smooths out seasonal esports spikes, revealing the true underlying growth trajectory. The algorithm successfully tags massive organic spikes as anomalies (red Xs).</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════
elif section == "6. Summary":
    st.subheader("Methodology & Executive Summary")

    df_sum = pd.DataFrame(
        [
            [
                "ADF Test",
                "Stationarity",
                "Players & Twitch: d=1 | Trends: d=0",
                "Valid",
            ],
            [
                "ARIMA(2,1,2)",
                "Univariate Forecast",
                "Forecasts plateau at ~960K/month",
                "Valid",
            ],
            [
                "VAR",
                "Multivariate Forecast",
                "Optimal lag: 5. Predicts cyclical recovery",
                "Valid",
            ],
            [
                "Granger Causality",
                "Predictive Power",
                "Trends → Players confirmed at lags 3-5",
                "Valid",
            ],
            [
                "GARCH(1,1)",
                "Volatility Modelling",
                "α+β=0.9999. Identifies long memory shocks",
                "Valid",
            ],
        ],
        columns=["Method", "Purpose", "Result", "Status"],
    )

    st.dataframe(
        df_sum.style.map(
            lambda x: "color: #3fb950; font-weight: bold;", subset=["Status"]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        '<div class="finding"><b>Google Trends is a leading indicator:</b> '
        "Declines in search interest provide a 3-5 month early-warning signal "
        "for player count softening.</div>"
        '<div class="finding"><b>Twitch viewership is structurally '
        "independent:</b> Despite rising during esports events, Twitch "
        "carries no predictive information about future active player "
        "counts.</div>"
        '<div class="finding"><b>Volatility is persistent:</b> '
        "GARCH estimation indicates that elevated "
        "uncertainty from major events "
        "dissipates very slowly over "
        "12–18 months.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Export Report Button
    csv_data = df_sum.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Summary Report (CSV)",
        data=csv_data,
        file_name="tsa_summary_report.csv",
        mime="text/csv",
    )
