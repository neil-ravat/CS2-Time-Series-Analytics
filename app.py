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

warnings.filterwarnings('ignore')

# Must be the first Streamlit command
st.set_page_config(
    page_title="CS2 Time Series",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── HACKING STREAMLIT: SaaS Mode CSS ──
st.markdown("""
<style>
@import url(
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
);

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
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
    background-color: #0d1117 !important;
    border-right: 1px solid #30363d;
}
.nav-section {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.3rem;
}

/* Stat Cards */
.stat-card {
    background: #161b22;
    padding: 1.2rem;
    border-radius: 8px;
    border: 1px solid #30363d;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: #58a6ff;
}
.stat-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.stat-val {
    font-size: 2rem;
    font-weight: 700;
    color: #c9d1d9;
    line-height: 1;
}
.stat-note {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.4rem;
}

/* Findings / Insights Boxes */
.finding {
    background: #1c2128;
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.95rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.finding b { color: #ffffff; font-weight: 600; }
.finding.key { border-left-color: #ff7b72; background: #2a191d; }

/* Analysis Text */
.analysis {
    font-size: 1rem;
    color: #8b949e;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {
        'models': {
            'arima': {'order': [2, 1, 2], 'steps': 12},
            'var': {'maxlags': 12, 'steps': 12},
            'garch': {'p': 1, 'q': 1},
            'cross_correlation': {'max_lags': 12},
            'acf_pacf': {'nlags': 20}
        }
    }

df, df_full, df_multi = get_data_splits()

# Plotly Theme Settings
PLOT_BG = 'rgba(0,0,0,0)'
PAPER_BG = 'rgba(0,0,0,0)'
GRID_COLOR = '#30363d'
TEXT_COLOR = '#c9d1d9'

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 📊 CS2 Analytics")
    st.markdown(
        '<div class="nav-section">Dashboard View</div>',
        unsafe_allow_html=True
    )
    section = st.radio("", [
        "1. EDA Overview",
        "2. Stationarity",
        "3. ARIMA Forecast",
        "4. VAR + Granger",
        "5. GARCH Volatility",
        "6. Summary"
    ], label_visibility="collapsed")

    st.markdown(
        '<div class="nav-section" style="margin-top:2rem">Data Specs</div>',
        unsafe_allow_html=True
    )
    st.caption(f"**Total Records:** {len(df)}")
    st.caption("**Time Horizon:** Jul 2012 — Apr 2026")
    st.caption(f"**Multivariate Overlap:** {len(df_multi)} months")

# ── PAGE HEADER ──
st.title("Counter-Strike 2 Engagement Ecosystem")
st.markdown(
    "*Multivariate Time Series Analysis · Steam & Twitch · 2012–2026*"
)
st.divider()

# ══════════════════════════════════════════
# 1. EDA OVERVIEW
# ══════════════════════════════════════════
if section == "1. EDA Overview":
    st.subheader("Exploratory Data Analysis")
    st.markdown(
        '<div class="analysis">Analyzing the three core dimensions of game '
        'engagement: Active participation (Steam), Public awareness (Google), '
        'and Spectatorship (Twitch).</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Players</div>'
        f'<div class="stat-val">{df["Players"].max()/1e6:.2f}M</div>'
        '<div class="stat-note">May 2023</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Trends Index</div>'
        f'<div class="stat-val">{int(df["Trends"].max())}</div>'
        '<div class="stat-note">Aug 2012</div></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">Peak Twitch Viewers'
        f'</div><div class="stat-val">{int(df_multi["Twitch"].max()/1e3)}K'
        '</div><div class="stat-note">Concurrent Avg</div></div>',
        unsafe_allow_html=True
    )

    # PLOTLY INTERACTIVE CHART
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            'Steam — Monthly Active Players',
            'Google Trends — Search Index',
            'Twitch — Concurrent Viewers'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_full.index, y=df_full['Players'], fill='tozeroy',
            name='Players', line=dict(color='#58a6ff', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index, y=df_full['Trends'], fill='tozeroy',
            name='Trends', line=dict(color='#3fb950', width=2)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_multi.index, y=df_multi['Twitch'], fill='tozeroy',
            name='Twitch', line=dict(color='#bc8cff', width=2)
        ),
        row=3, col=1
    )

    # Add CS2 release markers
    for i in range(1, 4):
        fig.add_vline(
            x='2023-09-01', line_width=1, line_dash="dash",
            line_color="#ff7b72", row=i, col=1
        )

    fig.update_layout(
        height=800, showlegend=False, paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>Structural break — CS2 release.</b> '
        'The September 2023 launch established a new baseline equilibrium '
        'near 1M concurrent players.</div>'
        '<div class="finding"><b>Divergence of Trends and Players.</b> '
        'Google search interest has declined since 2012, while actual player '
        'counts grew, reflecting a loyal, habitual user base.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Seasonal Decomposition (Players)")
    trend, seasonal, resid = models.get_seasonal_decomposition(df_full['Players'])
    fig_sd = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           subplot_titles=('Trend', 'Seasonal', 'Residual'))
    fig_sd.add_trace(go.Scatter(x=trend.index, y=trend, line=dict(color='#58a6ff')), row=1, col=1)
    fig_sd.add_trace(go.Scatter(x=seasonal.index, y=seasonal, line=dict(color='#3fb950')), row=2, col=1)
    fig_sd.add_trace(go.Scatter(x=resid.index, y=resid, line=dict(color='#ff7b72')), row=3, col=1)
    fig_sd.update_layout(height=500, showlegend=False, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR))
    fig_sd.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig_sd.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_sd, use_container_width=True)

# ══════════════════════════════════════════
# 2. STATIONARITY
# ══════════════════════════════════════════
elif section == "2. Stationarity":
    st.subheader("Augmented Dickey-Fuller Test")
    st.markdown(
        '<div class="analysis">Evaluating unit roots before modelling. '
        'Tests are conducted in levels and after first differencing '
        '(d=1).</div>',
        unsafe_allow_html=True
    )

    def format_adf_row(name, series):
        stat, pval, is_stat = models.get_adf_test_results(series)
        return (name, stat, pval)

    rows_orig = [
        format_adf_row("Players", df_full['Players']),
        format_adf_row("Trends", df_full['Trends']),
        format_adf_row("Twitch", df_multi['Twitch'])
    ]
    rows_diff = [
        format_adf_row("Players (d=1)", df_full['Players'].diff()),
        format_adf_row("Trends (d=1)", df_full['Trends'].diff()),
        format_adf_row("Twitch (d=1)", df_multi['Twitch'].diff())
    ]

    def render_adf(rows):
        df_adf = pd.DataFrame(
            rows, columns=['Variable', 'ADF Statistic', 'p-value']
        )
        df_adf['Result'] = df_adf['p-value'].apply(
            lambda x: 'Stationary' if x < 0.05 else 'Non-Stationary'
        )

        def style_fn(v):
            if v == 'Stationary':
                return 'color: #3fb950; font-weight: bold;'
            return 'color: #ff7b72; font-weight: bold;'

        st.dataframe(
            df_adf.style.map(style_fn, subset=['Result']),
            hide_index=True,
            use_container_width=True
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
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=('Δ Players', 'Δ Trends', 'Δ Twitch')
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index, y=df_full['Players'].diff(),
            line=dict(color='#58a6ff', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_full.index, y=df_full['Trends'].diff(),
            line=dict(color='#3fb950', width=1.5)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_multi.index, y=df_multi['Twitch'].diff(),
            line=dict(color='#bc8cff', width=1.5)
        ),
        row=3, col=1
    )

    for i in range(1, 4):
        fig.add_hline(
            y=0, line_width=1, line_color="#8b949e",
            opacity=0.5, row=i, col=1
        )

    fig.update_layout(
        height=600, showlegend=False, paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ACF & PACF (Differenced Players)")
    acf_vals, acf_conf, pacf_vals, pacf_conf = models.get_acf_pacf_values(
        df_full['Players'].diff(), nlags=config['models']['acf_pacf']['nlags']
    )
    fig_acf = make_subplots(rows=1, cols=2, subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)'))
    
    # ACF
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#58a6ff'), row=1, col=1)
    fig_acf.add_trace(go.Scatter(x=list(range(len(acf_vals))), y=acf_conf[:, 0] - acf_vals, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
    fig_acf.add_trace(go.Scatter(x=list(range(len(acf_vals))), y=acf_conf[:, 1] - acf_vals, mode='lines', fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=1)
    
    # PACF
    fig_acf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#3fb950'), row=1, col=2)
    fig_acf.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_conf[:, 0] - pacf_vals, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
    fig_acf.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_conf[:, 1] - pacf_vals, mode='lines', fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=1, col=2)
    
    fig_acf.update_layout(height=350, showlegend=False, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR))
    fig_acf.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig_acf.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_acf, use_container_width=True)

# ══════════════════════════════════════════
# 3. ARIMA FORECAST
# ══════════════════════════════════════════
elif section == "3. ARIMA Forecast":
    arima_order = tuple(config['models']['arima']['order'])
    arima_steps = config['models']['arima']['steps']
    st.subheader(f"ARIMA{arima_order} — Player Count Projections")

    with st.spinner("Optimizing ARIMA model..."):
        players = df_full['Players']
        fitted, fc_mean, fc_ci = models.get_arima_forecast(
            players, order=arima_order, steps=arima_steps
        )
        
        # Calculate in-sample metrics
        in_sample_pred = fitted.predict(start=1, end=len(players)-1)
        metrics = models.compute_forecast_metrics(players.iloc[1:], in_sample_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Model Architecture</div>'
        f'<div class="stat-val">ARIMA{arima_order}</div>'
        '<div class="stat-note">Selected via Min AIC</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">AIC Score</div>'
        f'<div class="stat-val">{fitted.aic:.1f}</div>'
        '<div class="stat-note">Goodness of fit</div></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">In-Sample MAPE</div>'
        f'<div class="stat-val">{metrics["MAPE"]:.2f}%</div>'
        '<div class="stat-note">Mean Abs Pct Error</div></div>',
        unsafe_allow_html=True
    )
    c4.markdown(
        '<div class="stat-card"><div class="stat-lbl">12-Mo Trajectory</div>'
        f'<div class="stat-val">{fc_mean.mean()/1e3:.0f}K</div>'
        '<div class="stat-note">Projected Baseline</div></div>',
        unsafe_allow_html=True
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=players.index, y=players, name="Actual Players",
            line=dict(color="#58a6ff", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_mean.index, y=fc_mean, name="Forecast",
            line=dict(color="#ff7b72", width=3, dash="dash")
        )
    )

    # Confidence Interval Shading
    concat_x = pd.concat([
        pd.Series(fc_ci.index), pd.Series(fc_ci.index[::-1])
    ])
    concat_y = pd.concat([fc_ci.iloc[:, 0], fc_ci.iloc[::-1, 1]])

    fig.add_trace(
        go.Scatter(
            x=concat_x,
            y=concat_y,
            fill='toself', fillcolor='rgba(255, 123, 114, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        )
    )

    fig.update_layout(
        height=450, title="Steam Player Count Forecast (12 Months)",
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR), hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding"><b>Forecast interpretation.</b> '
        'The model projects player counts stabilizing near 960K. '
        'This flat trajectory reflects ARIMA\'s reliance on recent momentum '
        'following the post-CS2 plateau.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════
# 4. VAR + GRANGER
# ══════════════════════════════════════════
elif section == "4. VAR + Granger":
    st.subheader("Multivariate Dynamics (VAR) & Causality")

    var_maxlags = config['models']['var']['maxlags']
    var_steps = config['models']['var']['steps']

    var_data = df_multi[['Players', 'Trends', 'Twitch']].diff().dropna()
    with st.spinner("Fitting VAR model..."):
        best_lag, vf, fc_df2 = models.get_var_forecast(
            var_data, maxlags=var_maxlags, steps=var_steps
        )

    afc = [df_multi['Players'].iloc[-1]]
    for d in fc_df2['Players']:
        afc.append(afc[-1] + d)

    fc_dates = pd.date_range(start=df_multi.index[-1], periods=13, freq='MS')

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Optimal Lag</div>'
        f'<div class="stat-val">L={best_lag}</div>'
        '<div class="stat-note">Months of look-back</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">System</div>'
        '<div class="stat-val">3-Var</div>'
        '<div class="stat-note">Players / Trends / Twitch</div></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">Data Window</div>'
        '<div class="stat-val">2017+</div>'
        '<div class="stat-note">Multivariate intersection</div></div>',
        unsafe_allow_html=True
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_multi.index, y=df_multi['Players'], name="Actual",
            line=dict(color="#58a6ff", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates, y=afc, name="VAR Forecast",
            line=dict(color="#ff7b72", width=3, dash="dash")
        )
    )
    fig.update_layout(
        height=450, title="System-Aware Player Forecast (VAR)",
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Granger Causality Inference")
    c1, c2 = st.columns(2)

    def style_fn(v):
        if v == 'Significant':
            return 'color: #3fb950; font-weight: bold;'
        return 'color: #8b949e;'

    with c1:
        st.markdown("**Does Search Volume (Trends) predict Players?**")
        df_trends = models.get_granger_causality(
            var_data[['Players', 'Trends']], maxlag=best_lag
        )
        st.dataframe(
            df_trends.style.map(style_fn, subset=['Signal']),
            hide_index=True, use_container_width=True
        )

    with c2:
        st.markdown("**Do Viewers (Twitch) predict Players?**")
        df_twitch = models.get_granger_causality(
            var_data[['Players', 'Twitch']], maxlag=best_lag
        )
        st.dataframe(
            df_twitch.style.map(style_fn, subset=['Signal']),
            hide_index=True, use_container_width=True
        )

    st.markdown("### Cross-Correlation")
    ccf_max_lags = config['models']['cross_correlation']['max_lags']
    df_ccf = models.get_cross_correlation(var_data['Trends'], var_data['Players'], max_lags=ccf_max_lags)
    fig_ccf = go.Figure(go.Bar(x=df_ccf['Lag'], y=df_ccf['Correlation'], marker_color='#58a6ff'))
    fig_ccf.update_layout(
        height=300, title="Cross-Correlation: Trends vs Players",
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR)
    )
    st.plotly_chart(fig_ccf, use_container_width=True)

# ══════════════════════════════════════════
# 5. GARCH VOLATILITY
# ══════════════════════════════════════════
elif section == "5. GARCH Volatility":
    garch_p = config['models']['garch']['p']
    garch_q = config['models']['garch']['q']
    st.subheader(f"GARCH({garch_p},{garch_q}) — Volatility Clustering")

    with st.spinner("Estimating Conditional Variance..."):
        pd_ = df_full['Players'].diff().dropna()
        vol, prm = models.get_garch_volatility(pd_, p=garch_p, q=garch_q)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="stat-card"><div class="stat-lbl">Alpha (α)</div>'
        f'<div class="stat-val">{prm["alpha[1]"]:.4f}</div>'
        '<div class="stat-note">Shock Reactivity</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        '<div class="stat-card"><div class="stat-lbl">Beta (β)</div>'
        f'<div class="stat-val">{prm["beta[1]"]:.4f}</div>'
        '<div class="stat-note">Volatility Persistence</div></div>',
        unsafe_allow_html=True
    )

    mem_factor = prm["alpha[1]"] + prm["beta[1]"]
    c3.markdown(
        '<div class="stat-card"><div class="stat-lbl">α + β</div>'
        f'<div class="stat-val">{mem_factor:.4f}</div>'
        '<div class="stat-note">Memory Factor (~1.0 is high)</div></div>',
        unsafe_allow_html=True
    )

    # PLOTLY INTERACTIVE CHART
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol.index, y=vol, fill='tozeroy', name="Volatility",
            line=dict(color="#ff7b72", width=2)
        )
    )
    fig.add_vline(
        x='2020-03-01', line_width=1, line_dash="dash",
        line_color="#8b949e", annotation_text="COVID-19"
    )
    fig.add_vline(
        x='2023-09-01', line_width=1, line_dash="dash",
        line_color="#8b949e", annotation_text="CS2 Release"
    )
    fig.update_layout(
        height=450, title="Conditional Volatility of Player Base",
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="finding key"><b>High persistence '
        '(α + β ≈ 1.0).</b> '
        'Shocks to variance decay extremely slowly. '
        'The uncertainty introduced '
        'by the CS2 transition represents a structural '
        'change with long memory.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════
elif section == "6. Summary":
    st.subheader("Methodology & Executive Summary")

    df_sum = pd.DataFrame([
        ["ADF Test", "Stationarity",
         "Players & Twitch: d=1 | Trends: d=0", "Valid"],
        ["ARIMA(2,1,2)", "Univariate Forecast",
         "Forecasts plateau at ~960K/month", "Valid"],
        ["VAR", "Multivariate Forecast",
         "Optimal lag: 5. Predicts cyclical recovery", "Valid"],
        ["Granger Causality", "Predictive Power",
         "Trends → Players confirmed at lags 3-5", "Valid"],
        ["GARCH(1,1)", "Volatility Modelling",
         "α+β=0.9999. Identifies long memory shocks", "Valid"]
    ], columns=['Method', 'Purpose', 'Result', 'Status'])

    st.dataframe(
        df_sum.style.map(
            lambda x: 'color: #3fb950; font-weight: bold;',
            subset=['Status']
        ),
        hide_index=True, use_container_width=True
    )

    st.markdown(
        '<div class="finding"><b>Google Trends is a leading indicator:</b> '
        'Declines in search interest provide a 3-5 month early-warning signal '
        'for player count softening.</div>'
        '<div class="finding"><b>Twitch viewership is structurally '
        'independent:</b> Despite rising during esports events, Twitch '
        'carries no predictive information about future active player '
        'counts.</div>'
        '<div class="finding"><b>Volatility is persistent:</b> '
        'GARCH estimation indicates that elevated '
        'uncertainty from major events '
        'dissipates very slowly over '
        '12–18 months.</div>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Export Report Button
    csv_data = df_sum.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Summary Report (CSV)",
        data=csv_data,
        file_name='tsa_summary_report.csv',
        mime='text/csv',
    )