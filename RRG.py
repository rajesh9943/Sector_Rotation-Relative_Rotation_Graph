import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
# Configure page
st.set_page_config(
    page_title="Sector Rotation Analysis",
    layout="centered"
)
# Apply fixed screen width for app (1440px)
st.markdown(
    f"""
    <style>
      .stAppViewContainer .stMain .stMainBlockContainer{{ max-width: 1440px; }}
    </style>
  """,
    unsafe_allow_html=True,
)
def fetch_data(symbols, period_days, interval="1d"):
    """Fetch data from Yahoo Finance with error handling"""
    data = {}
    failed_symbols = []
    # Calculate start date with dynamic buffer
    end_date = datetime.now()
    if interval == "1d":
        buffer_days = 100
        min_data_points = 20
    else: # "1wk"
        buffer_days = 250
        min_data_points = 10
    start_date = end_date - timedelta(days=period_days + buffer_days)
    for symbol in symbols:
        try:
            hist = yf.download(symbol, start_date, end_date, interval=interval, multi_level_index=False, progress=False)
            if len(hist) < min_data_points: # Minimum data requirement
                failed_symbols.append(symbol)
                continue
            data[symbol] = hist['Close']
        except Exception as e:
            failed_symbols.append(symbol)
            continue
    return data, failed_symbols
def calculate_relative_strength(price_data, benchmark_data, period):
    """Calculate relative strength vs benchmark"""
    # Ensure we have enough data
    min_length = min(len(price_data), len(benchmark_data))
    if min_length < period:
        return None, None
    # Align data by index
    aligned_data = pd.DataFrame({
        'price': price_data,
        'benchmark': benchmark_data
    }).dropna()
    if len(aligned_data) < period:
        return None, None
    # Calculate relative strength (sector/benchmark)
    relative_strength = aligned_data['price'] / aligned_data['benchmark']
    # Calculate momentum (rate of change) - kept for compatibility, though not used
    rs_momentum = relative_strength.pct_change(period).dropna()
    return relative_strength, rs_momentum
def calculate_jdk_rs_ratio(relative_strength, short_period=10, long_period=30):
    """Calculate JdK RS-Ratio similar to RRG methodology
  
    Official formula: RS-Ratio = (RS / MA(RS, long_period)) * 100
    where RS = Price_Sector / Price_Benchmark
  
    Note: Standard RRG uses 10 and 30 periods for daily data
    """
    if len(relative_strength) < long_period:
        return None
    # Calculate moving average of relative strength
    rs_ma = relative_strength.rolling(window=long_period).mean()
  
    # Normalize relative strength to 100
    # This shows if current RS is above (>100) or below (<100) its average
    rs_ratio = (relative_strength / rs_ma) * 100
    return rs_ratio
def calculate_jdk_rs_momentum(rs_ratio, short_period=10, long_period=20):
    """Calculate JdK RS-Momentum using SMA difference"""
    if rs_ratio is None or len(rs_ratio) < long_period:
        return None
    short_sma = rs_ratio.rolling(window=short_period).mean()
    long_sma = rs_ratio.rolling(window=long_period).mean()
    momentum = (short_sma - long_sma) # Raw difference
    return momentum
def get_quadrant_info(rs_ratio, rs_momentum):
    """Determine quadrant and provide info"""
    if rs_ratio > 100 and rs_momentum > 0:
        return "Leading", "green", "🚀"
    elif rs_ratio > 100 and rs_momentum < 0:
        return "Weakening", "orange", "📉"
    elif rs_ratio < 100 and rs_momentum < 0:
        return "Lagging", "red", "📊"
    else:
        return "Improving", "blue", "📈"
def create_animated_rrg_plot(results, animation_period_days, animation_speed=800, tail_length=10):
    """Create animated Relative Rotation Graph"""
    all_frames_data = []
    dates = []
    for symbol, data in results.items():
        if data['rs_ratio'] is not None and len(data['rs_ratio']) > 0:
            if len(dates) == 0:
                dates = data['rs_ratio'].index.tolist()
            else:
                dates = [d for d in dates if d in data['rs_ratio'].index]
    if len(dates) < 2:
        st.error("Not enough data points for animation")
        return None
    end_date = dates[-1]
    start_date = end_date - timedelta(days=animation_period_days)
    dates = [d for d in dates if d >= start_date]
    if len(dates) < 2:
        st.error(f"Not enough data points in the selected {animation_period_days} day period")
        return None
    step = max(1, len(dates) // 30) # Approx 30 points for smoothness
    sampled_dates = dates[::step]
    if dates[-1] not in sampled_dates:
        sampled_dates.append(dates[-1])
    all_rs_ratios = []
    all_rs_momentum = []
    for symbol, data in results.items():
        if data['rs_ratio'] is not None and data['rs_momentum'] is not None:
            all_rs_ratios.extend(data['rs_ratio'].dropna().values)
            all_rs_momentum.extend(data['rs_momentum'].dropna().values)
    x_min, x_max = min(all_rs_ratios), max(all_rs_ratios)
    y_min, y_max = min(all_rs_momentum), max(all_rs_momentum)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_center = 100
    x_data_range = max(x_max - 100, 100 - x_min)
    x_range = [x_center - x_data_range - x_padding, x_center + x_data_range + x_padding]
    y_range = [-2 if y_min > 0 else y_min - y_padding, 6 if y_max < 6 else y_max + y_padding]
    frames = []
    colors = px.colors.qualitative.Set2
    for frame_idx, date in enumerate(sampled_dates):
        frame_data = []
        for i, (symbol, data) in enumerate(results.items()):
            if data['rs_ratio'] is None or data['rs_momentum'] is None:
                continue
            try:
                rs_ratio = data['rs_ratio'].loc[:date]
                rs_momentum = data['rs_momentum'].loc[:date]
                if len(rs_ratio) == 0 or len(rs_momentum) == 0:
                    continue
                trail_length = min(tail_length, len(rs_ratio)) # Use tail_length
                trail_x = rs_ratio.tail(trail_length).values
                trail_y = rs_momentum.tail(trail_length).values
                current_x = rs_ratio.iloc[-1]
                current_y = rs_momentum.iloc[-1]
                color = colors[i % len(colors)]
                quadrant, _, _ = get_quadrant_info(current_x, current_y)
                frame_data.append(go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode='lines+markers',
                    line=dict(color=color, width=3, shape='spline'),
                    marker=dict(size=6, color=color),
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate=f'<b>{symbol} Trail</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>'
                ))
                frame_data.append(go.Scatter(
                    x=[current_x],
                    y=[current_y],
                    mode='markers+text',
                    marker=dict(size=20, color=color, line=dict(width=2, color='white')),
                    text=[symbol],
                    textposition="middle right",
                    textfont=dict(size=12, color='black'),
                    name=f'{symbol}',
                    showlegend=True,
                    hovertemplate=f'<b>{symbol}</b><br>RS-Ratio: {current_x:.2f}<br>RS-Momentum: {current_y:.2f}<br>Quadrant: {quadrant}<extra></extra>'
                ))
            except Exception as e:
                st.warning(f"Error in frame data for {symbol}: {str(e)}")
                continue
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(title_text=f"Sector Rotation - {date.strftime('%d-%m-%Y')}")
        ))
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
    fig.add_shape(type="rect", x0=100, y0=0, x1=x_range[1], y1=y_range[1], fillcolor="rgba(0,255,0,0.1)", line=dict(color="rgba(0,0,0,0)"), layer="below")
    fig.add_shape(type="rect", x0=100, y0=y_range[0], x1=x_range[1], y1=0, fillcolor="rgba(255,165,0,0.1)", line=dict(color="rgba(0,0,0,0)"), layer="below")
    fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=100, y1=0, fillcolor="rgba(255,0,0,0.1)", line=dict(color="rgba(0,0,0,0)"), layer="below")
    fig.add_shape(type="rect", x0=x_range[0], y0=0, x1=100, y1=y_range[1], fillcolor="rgba(0,0,255,0.1)", line=dict(color="rgba(0,0,0,0)"), layer="below")
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, layer="below")
    fig.add_vline(x=100, line_dash="dash", line_color="black", opacity=0.5, layer="below")
    leading_x = (100 + x_range[1]) / 2
    leading_y = y_range[1] * 0.9
    fig.add_annotation(x=leading_x, y=leading_y, text="Leading<br>(Hold Position)", showarrow=False, font=dict(size=14))
    weakening_x = (100 + x_range[1]) / 2
    weakening_y = y_range[0] * 0.9
    fig.add_annotation(x=weakening_x, y=weakening_y, text="Weakening<br>(Look to Sell)", showarrow=False, font=dict(size=14))
    lagging_x = (x_range[0] + 100) / 2
    lagging_y = y_range[0] * 0.9
    fig.add_annotation(x=lagging_x, y=lagging_y, text="Lagging<br>(Avoid)", showarrow=False, font=dict(size=14))
    improving_x = (x_range[0] + 100) / 2
    improving_y = y_range[1] * 0.9
    fig.add_annotation(x=improving_x, y=improving_y, text="Improving<br>(Look to Buy)", showarrow=False, font=dict(size=14))
    fig.update_layout(
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        width=800,
        height=1000,
        title=f"Sector Rotation - {sampled_dates[0].strftime('%d-%m-%Y')}" if sampled_dates else "Sector Rotation",
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': animation_speed, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': animation_speed // 2}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0,
            'xanchor': 'left',
            'y': -0.15,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': -0.2,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Date: ',
                'visible': True,
                'xanchor': 'center'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': sampled_dates[int(f.name)].strftime('%d-%m-%Y'),
                    'method': 'animate'
                }
                for f in frames
            ]
        }]
    )
    return fig
def create_static_rrg_plot(results, tail_length, show_tail=False):
    """Create static Relative Rotation Graph"""
    fig = go.Figure()
    # Calculate data ranges
    all_rs_ratios = []
    all_rs_momentum = []
    for symbol, data in results.items():
        if data['rs_ratio'] is not None and data['rs_momentum'] is not None:
            rs_ratio_vals = data['rs_ratio'].dropna().values
            rs_momentum_vals = data['rs_momentum'].dropna().values
            if len(rs_ratio_vals) > 0 and len(rs_momentum_vals) > 0:
                all_rs_ratios.extend(rs_ratio_vals)
                all_rs_momentum.extend(rs_momentum_vals)
    if not all_rs_ratios or not all_rs_momentum:
        st.error("No valid data to plot")
        return None
    x_min, x_max = min(all_rs_ratios), max(all_rs_ratios)
    y_min, y_max = min(all_rs_momentum), max(all_rs_momentum)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_center = 100
    x_data_range = max(x_max - 100, 100 - x_min)
    x_range = [x_center - x_data_range - x_padding, x_center + x_data_range + x_padding]
  
    if y_min > 0:
        y_range = [min(-0.5, y_min - y_padding), y_max + y_padding]
    elif y_max < 0:
        y_range = [y_min - y_padding, max(0.5, y_max + y_padding)]
    else:
        y_range = [y_min - y_padding, y_max + y_padding]
    # Add quadrant backgrounds
    fig.add_shape(
        type="rect",
        x0=100, y0=0, x1=x_range[1], y1=y_range[1],
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        layer="below"
    )
    fig.add_shape(
        type="rect",
        x0=100, y0=y_range[0], x1=x_range[1], y1=0,
        fillcolor="rgba(255,165,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        layer="below"
    )
    fig.add_shape(
        type="rect",
        x0=x_range[0], y0=y_range[0], x1=100, y1=0,
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        layer="below"
    )
    fig.add_shape(
        type="rect",
        x0=x_range[0], y0=0, x1=100, y1=y_range[1],
        fillcolor="rgba(0,0,255,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        layer="below"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, layer="below")
    fig.add_vline(x=100, line_dash="dash", line_color="black", opacity=0.5, layer="below")
    colors = px.colors.qualitative.Set2
    for i, (symbol, data) in enumerate(results.items()):
        if data['rs_ratio'] is None or data['rs_momentum'] is None:
            continue
        rs_ratio = data['rs_ratio'].dropna()
        rs_momentum = data['rs_momentum'].dropna()
        tail_points = min(tail_length, len(rs_ratio))
        if tail_points < 2:
            continue
        x_vals = rs_ratio.tail(tail_points).values
        y_vals = rs_momentum.tail(tail_points).values
        color = colors[i % len(colors)]
        if show_tail:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=color, width=3, shape='spline'),
                marker=dict(size=6, color=color),
                opacity=0.85,
                showlegend=False,
                hovertemplate=f'<b>{symbol} Trail</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>'
            ))
        current_quad, _, _ = get_quadrant_info(x_vals[-1], y_vals[-1])
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]],
            y=[y_vals[-1]],
            mode='markers+text',
            name=f'{symbol} ({current_quad})',
            marker=dict(size=20, color=color, line=dict(width=2, color='white')),
            text=[f'{symbol}'],
            textposition="middle right",
            textfont=dict(size=15, color='black'),
            hovertemplate=f'<b>{symbol}</b><br>RS-Ratio: {x_vals[-1]:.2f}<br>RS-Momentum: {y_vals[-1]:.2f}<br>Quadrant: {current_quad}<extra></extra>'
        ))
    fig.update_layout(
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        width=800,
        height=1000,
        showlegend=False
    )
    # Add quadrant labels
    leading_x = (100 + x_range[1]) / 2
    leading_y = y_range[1] * 0.8
    fig.add_annotation(x=leading_x, y=leading_y, text="Leading<br>(Hold Position)",
                       showarrow=False, font=dict(size=14))
    weakening_x = (100 + x_range[1]) / 2
    weakening_y = y_range[0] * 0.8
    fig.add_annotation(x=weakening_x, y=weakening_y, text="Weakening<br>(Look to Sell)",
                       showarrow=False, font=dict(size=14))
    lagging_x = (x_range[0] + 100) / 2
    lagging_y = y_range[0] * 0.8
    fig.add_annotation(x=lagging_x, y=lagging_y, text="Lagging<br>(Avoid)",
                       showarrow=False, font=dict(size=14))
    improving_x = (x_range[0] + 100) / 2
    improving_y = y_range[1] * 0.8
    fig.add_annotation(x=improving_x, y=improving_y, text="Improving<br>(Look to Buy)",
                       showarrow=False, font=dict(size=14))
    return fig
def main():
    st.subheader("Sector Rotation - Relative Rotation Graph")
    st.markdown(
        "Analyze sector/stock performance relative to benchmark using RRG methodology. Input Symbols as seen in Yahoo Finance")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        benchmark_map = {
            "Nifty 50": "^NSEI",
            "Nifty Bank": "^NSEBANK",
            "Nifty IT": "^CNXIT",
            "Nifty Pharma": "^CNXPHARMA",
            "Nifty FMCG": "^CNXFMCG",
            "Nifty Auto": "^CNXAUTO",
            "Nifty Metal": "^CNXMETAL",
            "Nifty Media": "^CNXMEDIA",
            "Nifty Realty": "^CNXREALTY",
            "Nifty Infrastructure": "^CNXINFRA",
            "Nifty Energy": "^CNXENERGY",
            "Nifty PSU Bank": "^CNXPSUBANK",
            "Nifty Public Sector Enterprises": "^CNXPSE",
            "Nifty Consumption": "^CNXCONSUM",
            "Nifty 100": "^CNX100",
            "Nifty 200": "^CNX200",
            "Nifty 500": "^CRSLDX",
            "Nifty Next 50 (Junior)": "^NSMIDCP"
        }
        benchmark_names = list(benchmark_map.keys())
        selected_benchmark_name = st.selectbox(
            "Benchmark Index",
            options=benchmark_names,
            index=benchmark_names.index("Nifty 50"),
            help="Select benchmark index (e.g., Nifty 50)"
        )
        benchmark = benchmark_map[selected_benchmark_name]
        period_options = {
            "1 Day": 1,
            "5 Days": 5,
            "1 Week": 7,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
        }
      
        selected_period = st.selectbox(
            "Analysis Period",
            options=list(period_options.keys()),
            index=0,
            help="Select the time period for analysis"
        )
      
        period = period_options[selected_period]
      
        # Time frame selection
        time_frame = st.selectbox(
            "Candle Timeframe",
            options=["Daily", "Weekly"],
            index=0,
            help="Select daily or weekly data granularity"
        )
        interval = "1d" if time_frame == "Daily" else "1wk"
        if time_frame == "Weekly" and period < 60:
            st.warning("For short analysis periods with weekly data, results may be limited due to fewer data points. Consider using a longer period or daily timeframe.")
      
        tail_length = st.slider("Tail Length (days)", min_value=2, max_value=30, value=10, step=1,
                            help="Number of historical points to show in the trail")
      
        enable_animation = st.checkbox("Enable Animation", value=False,
                                    help="Animate sector movement over time (shows how sectors rotate through quadrants)")
      
        if enable_animation:
            animation_speed = st.slider("Animation Speed (ms per frame)",
                                    min_value=200, max_value=2000, value=800, step=100,
                                    help="Control animation speed (lower = faster)")
    with col2:
        default_sectors = ["^NSEBANK", "^CNXIT", "^CNXPHARMA", "^CNXFMCG", "^CNXAUTO", "^CNXMETAL", "^CNXMEDIA", "^CNXREALTY", "^CNXINFRA", "^CNXENERGY", "^CNXPSUBANK", "^CNXPSE", "^CNXCONSUM", "^CNX100", "^CNX200", "^CRSLDX", "^NSMIDCP"]
        sectors_text = st.text_area(
            "Enter Sector/Stock symbols (one per line)",
            value="\n".join(default_sectors),
            height=220,
            help="Enter each sector/stock symbol on a new line"
        )
        sectors = [s.strip() for s in sectors_text.split('\n') if s.strip()]
        if not enable_animation:
            show_tail = st.checkbox(label="Show Tail", value=True, help="Display the historical trajectory of each sector")
    if st.button("Run Analysis", type="primary"):
        if not sectors:
            st.error("Please enter at least one sector symbol")
            return
        with st.spinner("Fetching data and calculating metrics..."):
            benchmark_data, benchmark_failed = fetch_data([benchmark], period, interval=interval)
            if benchmark not in benchmark_data:
                st.error(f"Could not fetch data for benchmark: {benchmark}")
                return
            sector_data, failed_sectors = fetch_data(sectors, period, interval=interval)
            if not sector_data:
                st.error("Could not fetch data for any sectors")
                return
            if failed_sectors:
                st.warning(f"Could not fetch data for: {', '.join(failed_sectors)}")
            benchmark_prices = benchmark_data[benchmark]
            available_len = len(benchmark_prices)
            if available_len < 10:
                st.error("Insufficient data points for meaningful RRG analysis.")
                return
            long_period = min(30, max(5, available_len // 3))
            short_period = min(10, max(2, available_len // 5))
            results = {}
            for symbol, prices in sector_data.items():
                try:
                    rel_strength, _ = calculate_relative_strength(prices, benchmark_prices, 1)
                    if rel_strength is None or len(rel_strength) < long_period:
                        continue
                    rs_ratio = calculate_jdk_rs_ratio(rel_strength, long_period=long_period)
                    if rs_ratio is None or len(rs_ratio) < short_period:
                        continue
                    rs_momentum = calculate_jdk_rs_momentum(rs_ratio, short_period=short_period, long_period=long_period)
                    if rs_momentum is None:
                        continue
                    results[symbol] = {
                        'rs_ratio': rs_ratio,
                        'rs_momentum': rs_momentum,
                        'relative_strength': rel_strength
                    }
                except Exception as e:
                    st.warning(f"Error calculating metrics for {symbol}: {str(e)}")
                    continue
            if not results:
                st.error("Could not calculate metrics for any sectors. Try a longer period or check symbol validity.")
                return
            # Create plot based on animation choice
            if enable_animation:
                st.info("🎬 Use the Play/Pause buttons and timeline slider below the chart to control the animation!")
                fig = create_animated_rrg_plot(results, period, animation_speed, tail_length)
            else:
                fig = create_static_rrg_plot(results, tail_length, show_tail)
          
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            # Summary table
            st.subheader("Current Relative Positions of Sector/Stock")
            summary_data = []
            for symbol, data in results.items():
                if data['rs_ratio'] is not None and data['rs_momentum'] is not None:
                    current_ratio = data['rs_ratio'].iloc[-1] if len(data['rs_ratio']) > 0 else 0
                    current_momentum = data['rs_momentum'].iloc[-1] if len(data['rs_momentum']) > 0 else 0
                    quadrant, color, icon = get_quadrant_info(current_ratio, current_momentum)
                    summary_data.append({
                        'Sector': symbol,
                        'RS-Ratio': f"{current_ratio:.2f}",
                        'RS-Momentum': f"{current_momentum:.2f}",
                        'Quadrant': f"{quadrant}"
                    })
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            with st.expander("Understanding the Relative Rotation Graph"):
                st.markdown("""
                    **Quadrants Explanation:**
                    **Leading (Top-Right)**: High relative strength, positive momentum
                    - Sectors out performing benchmark with increasing momentum
                    **Weakening (Bottom-Right)**: High relative strength, negative momentum
                    - Sectors still out performing but losing momentum
                    **Lagging (Bottom-Left)**: Low relative strength, negative momentum
                    - Sectors under performing benchmark with decreasing momentum
                    **Improving (Top-Left)**: Low relative strength, positive momentum
                    - Sectors under performing but gaining momentum
                    **How to Read:**
                    - **RS-Ratio > 100**: Sector out performing benchmark
                    - **RS-Ratio < 100**: Sector under performing benchmark
                    - **RS-Momentum > 0**: Relative strength is improving
                    - **RS-Momentum < 0**: Relative strength is declining
                    - **Animation**: Shows how sectors move through quadrants over time
                    - **Tail**: Shows the trajectory of sector movement
                  
                    ---
                  
                    ### Daily vs Weekly Timeframes
                  
                    **Daily and weekly timeframes on RRG charts are used to distinguish between short-term and long-term relative strength trends.**
                  
                    #### Weekly RRG
                    **Use:** Identify major, long-term trends in relative performance between different assets or sectors.
                  
                    **View:** Provides the "big picture" and is suitable for long-term investors or swing traders with a time horizon of weeks to months.
                  
                    **Behavior:** Shows gradual rotations through the quadrants, as full rotations may take several weeks.
                  
                    #### Daily RRG
                    **Use:** Spot shorter-term tactical shifts, emerging momentum, or potential early warning signs of a trend change.
                  
                    **View:** Offers a more granular, reactive view of performance, ideal for short-term traders.
                  
                    **Behavior:** Shows faster, more rapid rotations through the quadrants, sometimes within just a few days.
                  
                    #### How to Use Them Together
                  
                    **Compare the two:** Look at the daily RRG for short-term strength or weakness while the weekly RRG confirms the longer-term trend.
                  
                    **Avoid getting "faked out":**
                    - A sector may look weak on the daily chart but is still leading on the weekly chart, indicating a temporary pullback rather than a full reversal.
                    - Conversely, a daily chart might show a sector entering the leading quadrant, but the weekly chart could show the long-term trend is still in decline, suggesting the short-term move is likely temporary.
                  
                    **Align timeframes:** Ensure the RRG timeframe you are using aligns with the timeframe of your regular price charts for consistency.
                  
                    **💡 Pro Tip:** The weekly RRG provides the "big picture" view of major leadership shifts for longer-term trends, while the daily RRG captures short-term dynamics, showing quicker rotations and emerging opportunities. Combining both timeframes is ideal for a comprehensive analysis, allowing you to see short-term movements within a larger long-term trend.
                    """)
if __name__ == "__main__":
    main()
