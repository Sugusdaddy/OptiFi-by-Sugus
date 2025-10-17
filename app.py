import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
import plotly.graph_objects as go
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

st.markdown("""
<style>
    :root {
        --primary-color: #F8FAFC;
        --secondary-color: #FFFFFF;
        --accent-color: #0084FF;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--primary-color);
        color: #1F2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    .main {
        background-color: var(--primary-color);
        padding: 2rem;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        color: #0084FF;
    }
    
    h2 {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #1F2937;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
    }
    
    p {
        color: #4B5563;
    }
    
    [data-testid="stTabs"] [aria-selected="true"] {
        border-bottom: 3px solid #0084FF !important;
    }
    
    [data-testid="stTabs"] button {
        color: #6B7280;
        font-weight: 500;
        background-color: transparent !important;
    }
    
    [data-testid="stTabs"] button:hover {
        color: #0084FF;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0084FF;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #6B7280;
    }
    
    button[kind="primary"] {
        background: linear-gradient(90deg, #0084FF, #0066CC) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 132, 255, 0.3) !important;
    }
    
    [data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stSlider"] {
        background-color: var(--secondary-color) !important;
    }
    
    .stDataFrame {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
    }
    
    [data-testid="stInfo"] {
        background-color: #E0F2FE !important;
        border-left: 4px solid #0084FF !important;
        color: #0C4A6E !important;
    }
    
    [data-testid="stSuccess"] {
        background-color: #DCFCE7 !important;
        border-left: 4px solid #10B981 !important;
        color: #065F46 !important;
    }
    
    [data-testid="stError"] {
        background-color: #FEE2E2 !important;
        border-left: 4px solid #EF4444 !important;
        color: #7F1D1D !important;
    }
</style>
""", unsafe_allow_html=True)

ASSET_DATABASE = {
    "Technology": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "META": "Meta",
        "AMZN": "Amazon",
        "AMD": "AMD",
        "INTC": "Intel",
        "CSCO": "Cisco"
    },
    "Finance": {
        "JPM": "JPMorgan",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "BLK": "BlackRock",
        "SCHW": "Schwab",
        "ICE": "Intercontinental",
        "CME": "CME Group"
    },
    "Healthcare": {
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth",
        "PFE": "Pfizer",
        "MRK": "Merck",
        "ABBV": "AbbVie",
        "TMO": "Thermo Fisher",
        "LLY": "Eli Lilly",
        "BMY": "Bristol Myers"
    },
    "Energy": {
        "XOM": "ExxonMobil",
        "CVX": "Chevron",
        "COP": "ConocoPhillips",
        "SLB": "Schlumberger",
        "MPC": "Marathon",
        "PSX": "Phillips 66",
        "EOG": "EOG Resources"
    },
    "Consumer": {
        "WMT": "Walmart",
        "TM": "Toyota",
        "NKE": "Nike",
        "SBUX": "Starbucks",
        "HD": "Home Depot",
        "LOW": "Lowes",
        "COST": "Costco",
        "PG": "Procter Gamble"
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "XRP-USD": "Ripple",
        "ADA-USD": "Cardano",
        "DOGE-USD": "Dogecoin",
        "MATIC-USD": "Polygon",
        "LINK-USD": "Chainlink"
    }
}

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.prices = None
        self.returns = None
        self.ef = None
    
    def select_best_assets(self, num_assets=8):
        """Select best performing assets based on Sharpe ratio and volatility"""
        candidates = {}
        
        for category, assets in ASSET_DATABASE.items():
            for ticker, name in assets.items():
                try:
                    data = yf.download(ticker, period='2y', progress=False, interval='1d')
                    if data.empty or len(data) < 100:
                        continue
                    
                    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                    returns = prices.pct_change().dropna()
                    
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
                    
                    candidates[ticker] = {
                        'name': name,
                        'sharpe': sharpe,
                        'return': annual_return,
                        'volatility': annual_vol
                    }
                except:
                    continue
        
        if not candidates:
            return list(ASSET_DATABASE['Technology'].keys())[:5]
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        best_tickers = [ticker for ticker, _ in sorted_candidates[:num_assets]]
        
        return best_tickers
        
    def fetch_data(self, tickers, period='2y'):
        prices_dict = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                data = yf.download(ticker, period=period, progress=False, interval='1d')
                if not data.empty:
                    if 'Adj Close' in data.columns:
                        prices_dict[ticker] = data['Adj Close']
                    elif 'Close' in data.columns:
                        prices_dict[ticker] = data['Close']
            except:
                continue
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        
        if not prices_dict:
            return None
        
        self.prices = pd.concat(prices_dict, axis=1)
        self.prices.columns = list(prices_dict.keys())
        self.prices = self.prices.dropna()
        self.returns = self.prices.pct_change().dropna()
        
        self.prices.to_csv('data/prices_latest.csv')
        self.returns.to_csv('data/returns_latest.csv')
        
        return self.prices
    
    def optimize_portfolio(self, constraint_min=0.05):
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        self.ef = EfficientFrontier(
            mean_returns, 
            cov_matrix, 
            weight_bounds=(constraint_min, 0.5)
        )
        
        try:
            weights = self.ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        except:
            weights = self.ef.min_volatility()
        
        return weights
    
    def get_portfolio_stats(self):
        ret, vol, sharpe = self.ef.portfolio_performance(verbose=False)
        return ret, vol, sharpe
    
    def get_correlation_matrix(self):
        return self.returns.corr()
    
    def get_efficient_frontier_data(self, num_portfolios=2000):
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(len(self.returns.columns))
            w /= np.sum(w)
            
            ret = np.sum(mean_returns * w)
            std = np.sqrt(np.dot(w, cov_matrix @ w))
            sharpe = (ret - self.risk_free_rate) / std
            
            results[0,i] = std
            results[1,i] = ret
            results[2,i] = sharpe
        
        return results
    
    def allocate_capital(self, weights, capital=10000):
        allocation = DiscreteAllocation(weights, self.prices.iloc[-1], total_portfolio_value=capital)
        discrete_allocation, leftover = allocation.greedy_portfolio()
        return discrete_allocation, leftover

with st.sidebar:
    st.markdown("<h2>Configuration</h2>", unsafe_allow_html=True)
    st.divider()
    
    mode = st.radio("Mode", ["Manual Selection", "AI Selection"], horizontal=True)
    
    if mode == "Manual Selection":
        st.markdown("<h3>Select Assets</h3>", unsafe_allow_html=True)
        
        selected_assets = []
        for category, assets in ASSET_DATABASE.items():
            with st.expander(category, expanded=(category == "Technology")):
                for ticker, name in assets.items():
                    if st.checkbox(name, key=ticker):
                        selected_assets.append(ticker)
        
        if not selected_assets:
            selected_assets = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
            st.warning("No assets selected. Using default portfolio.")
    
    else:
        st.markdown("<h3>AI Asset Selection</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem; color: #6B7280;'>The AI will analyze all available assets and select the best performers.</p>", unsafe_allow_html=True)
        
        num_assets = st.slider("Number of Assets to Select", 5, 10, 8, label_visibility="collapsed")
        
        if st.button("Analyze and Select Best Assets", use_container_width=True, key="ai_select"):
            st.session_state.ai_selecting = True
        
        if st.session_state.get('ai_selecting', False):
            with st.spinner("Analyzing all assets..."):
                optimizer_temp = PortfolioOptimizer(risk_free_rate=0.05)
                selected_assets = optimizer_temp.select_best_assets(num_assets=num_assets)
                st.session_state.ai_selected_assets = selected_assets
                st.session_state.ai_selecting = False
        
        selected_assets = st.session_state.get('ai_selected_assets', ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD'])
        
        with st.expander("Selected Assets", expanded=True):
            for asset in selected_assets:
                asset_name = None
                for category, assets in ASSET_DATABASE.items():
                    if asset in assets:
                        asset_name = assets[asset]
                        break
                st.markdown(f"✓ {asset}: {asset_name}")
    
    st.divider()
    
    st.markdown("<h3>Parameters</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.85rem; color: #6B7280; margin-bottom: 0.3rem;'>Period</p>", unsafe_allow_html=True)
    period = st.selectbox(
        "Historical Period",
        ["1y", "2y", "5y", "10y"],
        index=1,
        label_visibility="collapsed"
    )
    
    st.markdown("<p style='font-size: 0.85rem; color: #6B7280; margin-bottom: 0.3rem; margin-top: 1rem;'>Capital</p>", unsafe_allow_html=True)
    capital = st.number_input(
        "Capital to Invest",
        min_value=1000,
        value=10000,
        step=1000,
        format="%d",
        label_visibility="collapsed"
    )
    
    st.markdown("<p style='font-size: 0.85rem; color: #6B7280; margin-bottom: 0.3rem; margin-top: 1rem;'>Risk-Free Rate: <span style='color: #0084FF;'>5.0%</span></p>", unsafe_allow_html=True)
    rf_rate = st.slider(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        label_visibility="collapsed"
    ) / 100
    
    st.markdown("<p style='font-size: 0.85rem; color: #6B7280; margin-bottom: 0.3rem; margin-top: 1rem;'>Min Weight: <span style='color: #0084FF;'>5.0%</span></p>", unsafe_allow_html=True)
    min_weight = st.slider(
        "Minimum Weight per Asset",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        label_visibility="collapsed"
    ) / 100
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        optimize_btn = st.button("Optimize", use_container_width=True, key="opt_btn")
    with col2:
        clear_btn = st.button("Clear", use_container_width=True, key="clear_btn")
    
    if clear_btn:
        st.session_state.optimize = False
        for f in ['data/prices_latest.csv', 'data/returns_latest.csv']:
            if os.path.exists(f):
                os.remove(f)
        st.rerun()
    
    if optimize_btn:
        st.session_state.optimize = True
    
    if 'optimize' not in st.session_state:
        st.session_state.optimize = False

st.markdown("<h1>Portfolio Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #0084FF; font-size: 0.95rem; margin-bottom: 2rem;'>Modern Portfolio Theory | Multi-Asset Optimization</p>", unsafe_allow_html=True)
st.divider()

if st.session_state.optimize:
    with st.spinner("Processing data and optimizing..."):
        optimizer = PortfolioOptimizer(risk_free_rate=rf_rate)
        prices = optimizer.fetch_data(selected_assets, period=period)
        
        if prices is not None:
            optimal_weights = optimizer.optimize_portfolio(constraint_min=min_weight)
            ret, vol, sharpe = optimizer.get_portfolio_stats()
            
            st.markdown("<h2>Portfolio Summary</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Return", f"{ret*100:.2f}%", f"+{(ret-rf_rate)*100:.1f}%")
            with col2:
                st.metric("Volatility", f"{vol*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            with col4:
                st.metric("Risk-Free Rate", f"{rf_rate*100:.1f}%")
            
            st.divider()
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Composition", "Efficient Frontier", "Correlations", "Capital Allocation", "Historical", "Risk Metrics"
            ])
            
            with tab1:
                weights_df = pd.DataFrame({
                    'Asset': list(optimal_weights.keys()),
                    'Weight': [f"{w*100:.2f}%" for w in optimal_weights.values()],
                    'Amount': [f"${w*capital:,.0f}" for w in optimal_weights.values()]
                }).sort_values('Weight', ascending=False)
                
                col_table, col_pie = st.columns([1.3, 1])
                
                with col_table:
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
                
                with col_pie:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(optimal_weights.keys()),
                        values=[w*100 for w in optimal_weights.values()],
                        marker=dict(colors=['#0084FF', '#0099FF', '#FF0066', '#FFAA00', '#10B981', '#FF6600', '#00AAFF', '#FF3366']),
                    )])
                    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1F2937'))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv = weights_df.to_csv(index=False)
                    st.download_button("Download Portfolio CSV", csv, f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
                
                with col_d2:
                    summary_text = f"Portfolio Summary\n\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\nExpected Return: {ret*100:.2f}%\nVolatility: {vol*100:.2f}%\nSharpe Ratio: {sharpe:.3f}\n\n{weights_df.to_string()}"
                    st.download_button("Download Summary TXT", summary_text, f"summary_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain", use_container_width=True)
            
            with tab2:
                frontier_data = optimizer.get_efficient_frontier_data(num_portfolios=3000)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frontier_data[0]*100, y=frontier_data[1]*100,
                    mode='markers',
                    marker=dict(size=5, color=frontier_data[2], colorscale='Viridis', showscale=True, opacity=0.6),
                    name='Portfolios'
                ))
                fig.add_trace(go.Scatter(
                    x=[vol*100], y=[ret*100],
                    mode='markers+text',
                    marker=dict(size=25, color='red', symbol='star', line=dict(color='white', width=2)),
                    text=[f'Optimal\nSharpe: {sharpe:.3f}'],
                    textposition='top center',
                    name='Optimal Portfolio'
                ))
                
                fig.update_layout(
                    title='Efficient Frontier: Mean-Variance Optimization',
                    xaxis_title='Risk (Volatility %)',
                    yaxis_title='Expected Return (%)',
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1F2937'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                fig.write_html("plots/efficient_frontier.html")
                st.success("Efficient frontier saved")
            
            with tab3:
                corr_matrix = optimizer.get_correlation_matrix()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}'
                ))
                fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1F2937'))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                discrete_alloc, leftover = optimizer.allocate_capital(optimal_weights, capital)
                
                alloc_data = []
                total_invested = 0
                
                for ticker, shares in sorted(discrete_alloc.items()):
                    price = optimizer.prices[ticker].iloc[-1]
                    value = shares * price
                    total_invested += value
                    alloc_data.append({
                        'Asset': ticker,
                        'Shares': int(shares),
                        'Price': f"${price:,.2f}",
                        'Value': f"${value:,.0f}",
                        'Allocation': f"{(value/capital)*100:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(alloc_data), use_container_width=True, hide_index=True)
                
                st.divider()
                
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.metric("Total Invested", f"${total_invested:,.0f}")
                with col_k2:
                    st.metric("Cash Remaining", f"${leftover:,.0f}")
                with col_k3:
                    st.metric("Capital Utilization", f"{(total_invested/capital)*100:.1f}%")
            
            with tab5:
                normalized_prices = optimizer.prices / optimizer.prices.iloc[0] * 100
                
                fig = go.Figure()
                for col in normalized_prices.columns:
                    fig.add_trace(go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices[col],
                        mode='lines',
                        name=col,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Normalized Price Performance (Base 100)',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(248,250,252,0.5)',
                    font=dict(color='#1F2937'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                fig.write_html("plots/price_performance.html")
            
            with tab6:
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown("<h3>Individual Asset Metrics</h3>", unsafe_allow_html=True)
                    metrics_data = []
                    for ticker in optimizer.returns.columns:
                        ann_ret = optimizer.returns[ticker].mean() * 252 * 100
                        ann_vol = optimizer.returns[ticker].std() * np.sqrt(252) * 100
                        sharpe_ind = (ann_ret/100 - rf_rate) / (ann_vol/100)
                        metrics_data.append({
                            'Asset': ticker,
                            'Return': f"{ann_ret:.1f}%",
                            'Volatility': f"{ann_vol:.1f}%",
                            'Sharpe': f"{sharpe_ind:.2f}"
                        })
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
                
                with col_m2:
                    st.markdown("<h3>Portfolio Statistics</h3>", unsafe_allow_html=True)
                    var_95 = float((optimizer.returns.mean() * 252 - 1.645 * optimizer.returns.std() * np.sqrt(252)).mean())
                    
                    st.markdown(f"""
                    **Annual Return:** {ret*100:.2f}%
                    
                    **Annual Volatility:** {vol*100:.2f}%
                    
                    **Sharpe Ratio:** {sharpe:.3f}
                    
                    **Value at Risk (95%):** {var_95*100:.2f}%
                    
                    **Asset Count:** {len(optimizer.returns.columns)}
                    
                    **Data Points:** {len(optimizer.prices)}
                    """)
        
        else:
            st.error("Error downloading data.")

else:
    st.info("Select assets from the sidebar and click Optimize to begin.")