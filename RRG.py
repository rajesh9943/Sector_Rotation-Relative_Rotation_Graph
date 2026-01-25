import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Sector Rotation - Relative Rotation Graph", layout="wide")

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.main .block-container {max-width: 1400px; padding-top: 1rem; padding-left: 2rem; padding-right: 2rem;}
.stButton > button {background-color: #ff4b4b; color: white; border-radius: 8px; height: 3em; width: 100%; font-size: 18px;}
div[data-testid="stNumberInput"] > div > div > input {font-size: 18px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SECTOR MAPPING
# -------------------------------------------------
SECTOR_MAPPING = {
    "Bank": "^NSEBANK", "IT": "^CNXIT", "Pharma": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "Auto": "^CNXAUTO", "Metal": "^CNXMETAL", "Media": "^CNXMEDIA", "Realty": "^CNXREALTY",
    "Infra": "^CNXINFRA", "Energy": "^CNXENERGY", "PSU Bank": "^CNXPSUBANK",
    "PSE": "^CNXPSE", "Consumption": "^CNXCONSUM", "Nifty Next 50":"^NSMIDCP",
    "Nifty 100": "^CNX100", "Nifty 200": "^CNX200", "Nifty 500": "^CRSLDX",
}
DEFAULT_SECTORS = list(SECTOR_MAPPING.keys())

# -------------------------------------------------
# DATA FETCH
# -------------------------------------------------
def fetch_data(symbols, period_days):
    data, failed = {}, []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 60)
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(df) < 30:
                failed.append(symbol)
                continue
            data[symbol] = df['Close']
        except:
            failed.append(symbol)
    return data, failed

# -------------------------------------------------
# CALCULATIONS
# -------------------------------------------------
def calculate_jdk_rs_ratio(rs_series):
    benchmark = rs_series.rolling(window=100, min_periods=20).mean()
    return (rs_series / benchmark) * 100

def calculate_jdk_rs_momentum(rs_ratio, period=8):
    return rs_ratio.pct_change(periods=period) * 100

def get_quadrant(rs_ratio, rs_momentum):
    if rs_ratio > 100 and rs_momentum > 0: return "Leading", "(Hold Position)"
    elif rs_ratio > 100 and rs_momentum <= 0: return "Weakening", "(Look to Sell)"
    elif rs_ratio <= 100 and rs_momentum <= 0: return "Lagging", "(Avoid)"
    else: return "Improving", "(Look to Buy)"

# -------------------------------------------------
# RRG PLOT (UPDATED: NO EXTERNAL dd/mm LABELS)
# -------------------------------------------------
def create_rrg_plot(results, tail_length, show_tail, display_names=None):
    fig = go.Figure()
    latest = [(d['rs_ratio'].iloc[-1], d['rs_momentum'].iloc[-1])
              for d in results.values() if not pd.isna(d['rs_ratio'].iloc[-1])]
    if not latest:
        st.error("No data available to plot.")
        return None

    xs, ys = zip(*latest)
    pad_x, pad_y = 10, 6
    x_range = [min(xs)-pad_x, max(xs)+pad_x]
    y_range = [min(ys)-pad_y, max(ys)+pad_y]

    # Quadrant backgrounds
    fig.add_shape(type="rect", x0=100, y0=0, x1=x_range[1], y1=y_range[1], fillcolor="rgba(220,255,220,0.5)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=y_range[0], x1=x_range[1], y1=0, fillcolor="rgba(255,230,180,0.5)", line_width=0)
    fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=100, y1=0, fillcolor="rgba(255,200,200,0.5)", line_width=0)
    fig.add_shape(type="rect", x0=x_range[0], y0=0, x1=100, y1=y_range[1], fillcolor="rgba(220,220,255,0.5)", line_width=0)

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=100, line_dash="dash", line_color="gray")

    colors = px.colors.qualitative.Plotly
    for i, (symbol, data) in enumerate(results.items()):
        ratio = data['rs_ratio'].dropna()
        momentum = data['rs_momentum'].dropna()
        dates = ratio.index
        tail = min(tail_length, len(ratio))
        x_tail, y_tail = ratio.tail(tail).values, momentum.tail(tail).values
        date_tail = dates[-tail:]
        color = colors[i % len(colors)]
        clean_name = display_names.get(symbol, symbol)

        if show_tail and tail > 1:
            # Tail line
            fig.add_trace(go.Scatter(
                x=x_tail, y=y_tail,
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))

            # Intermediate points on tail (small, semi-transparent) - for visual flow only
            for j in range(tail - 1):
                hover_date = date_tail[j].strftime("%d/%m/%y")
                fig.add_trace(go.Scatter(
                    x=[x_tail[j]], y=[y_tail[j]],
                    mode='markers',
                    marker=dict(size=7, color=color, opacity=0.5, line=dict(width=1, color='white')),
                    hovertemplate=(
                        f"<b>{clean_name}({symbol})</b><br>"
                        f"Date: {hover_date}<br>"
                        f"RS-Ratio: {x_tail[j]:.2f}<br>"
                        f"RS-Momentum: {y_tail[j]:.2f}<extra></extra>"
                    ),
                    showlegend=False
                ))
                # NO EXTERNAL DATE LABELS ANYMORE (removed as requested)

        # Latest point - large marker with sector name
        curr_x, curr_y = x_tail[-1], y_tail[-1]
        latest_date = date_tail[-1].strftime("%d/%m/%y")
        quadrant, subtitle = get_quadrant(curr_x, curr_y)

        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(width=2, color='white')),
            text=clean_name,
            textposition="middle right",
            textfont=dict(size=14, color="black"),
            hovertemplate=(
                f"<b>{clean_name}({symbol})</b><br>"
                f"Date: {latest_date}<br>"
                f"RS-Ratio: {curr_x:.2f}<br>"
                f"RS-Momentum: {curr_y:.2f}<br>"
                f"<b>{quadrant}</b> {subtitle}<extra></extra>"
            ),
            showlegend=False
        ))

    # Quadrant labels
    fig.add_annotation(x=(100 + x_range[1])/2, y=y_range[1]*0.9,
                       text="Leading<br>(Hold Position)", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(100 + x_range[1])/2, y=y_range[0]*0.9,
                       text="Weakening<br>(Look to Sell)", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(x_range[0] + 100)/2, y=y_range[0]*0.9,
                       text="Lagging<br>(Avoid)", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(x_range[0] + 100)/2, y=y_range[1]*0.9,
                       text="Improving<br>(Look to Buy)", showarrow=False, font=dict(size=14))

    fig.update_layout(
        title="Relative Rotation Graph (RRG)",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        plot_bgcolor="white",
        hovermode="closest",
        height=800,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("Sector Rotation - Relative Rotation Graph")
st.markdown("Analyze sector/stock rotation relative to a benchmark using RRG methodology.")

col1, col2 = st.columns(2)

with col1:
    st.text_input("Benchmark Symbol", value="Nifty 50", key="benchmark_input")
    
    st.markdown("**Analysis Period (days)**")
    period_days = st.number_input("Analysis Period (days)", 1, 1826, 183, label_visibility="collapsed")
    
    st.markdown("**Tail Length (points)**")
    tail_length = st.number_input("Tail Length (points)", 1, 30, 5, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("Run Analysis", type="primary")

with col2:
    st.text_area("Enter Sector/Stock symbols (one per line)", "\n".join(DEFAULT_SECTORS), height=300, key="symbols_input")
    show_tail = st.checkbox("Show Tail", value=True)

# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
if run_button:
    benchmark_map = {"Nifty 50": "^NSEI", "Sensex": "^BSESN", **SECTOR_MAPPING}
    benchmark = benchmark_map.get(st.session_state.benchmark_input, st.session_state.benchmark_input)
    
    input_list = [s.strip() for s in st.session_state.symbols_input.split("\n") if s.strip()]
    stocks, display_names = [], {}
    for item in input_list:
        if item in SECTOR_MAPPING:
            stocks.append(SECTOR_MAPPING[item])
            display_names[SECTOR_MAPPING[item]] = item
        else:
            stocks.append(item)
            display_names[item] = item

    with st.spinner("Fetching data and calculating Relative Rotation..."):
        data, failed = fetch_data([benchmark] + stocks, period_days)
        benchmark_series = data.get(benchmark)
        
        if benchmark_series is None:
            st.error("Failed to fetch benchmark data.")
        else:
            if failed:
                st.warning(f"Failed to fetch: {', '.join(failed)}")
            
            results = {}
            for symbol in stocks:
                if symbol not in data:
                    continue
                aligned = pd.concat([data[symbol], benchmark_series], axis=1).dropna()
                if len(aligned) < 30:
                    continue
                rs = aligned.iloc[:, 0] / aligned.iloc[:, 1]
                rs_ratio = calculate_jdk_rs_ratio(rs)
                results[symbol] = {
                    'rs_ratio': rs_ratio,
                    'rs_momentum': calculate_jdk_rs_momentum(rs_ratio)
                }

            fig = create_rrg_plot(results, tail_length, show_tail, display_names)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Current Relative Positions")
                table = []
                for sym, d in results.items():
                    r = d['rs_ratio'].iloc[-1]
                    m = d['rs_momentum'].iloc[-1]
                    q, adv = get_quadrant(r, m)
                    clean = display_names.get(sym, sym.replace("^", "").replace("CNX", "").replace("NSE", ""))
                    table.append({
                        "Sector": clean,
                        "RS-Ratio": f"{r:.2f}",
                        "RS-Momentum": f"{m:.2f}",
                        "Quadrant": q,
                        "Advice": adv
                    })
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            else:
                st.error("No valid data to display.")