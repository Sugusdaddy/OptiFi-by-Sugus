# Portfolio Optimizer

Advanced multi-asset portfolio optimization platform using Modern Portfolio Theory (Markowitz). Combines traditional equities, cryptocurrencies, and commodities to construct optimal portfolios with maximum risk-adjusted returns.

## Overview

This application implements quantitative portfolio optimization techniques to help investors construct diversified portfolios. Using historical market data and mathematical optimization, it identifies the portfolio composition that maximizes the Sharpe ratio—the excess return per unit of risk.

## Key Features

- **Multi-Asset Class Support**: Stocks, cryptocurrencies, commodities, and indices
- **AI-Powered Asset Selection**: Automatic selection of best-performing assets based on Sharpe ratio analysis
- **Efficient Frontier Visualization**: Interactive 3000-portfolio Monte Carlo simulation
- **Sharpe Ratio Maximization**: Constrained optimization for optimal risk-return balance
- **Correlation Analysis**: Asset correlation heatmaps for diversification insights
- **Capital Allocation Engine**: Discrete share recommendations with exact dollar amounts
- **Comprehensive Risk Metrics**: VaR, CVaR, volatility, and individual asset analysis
- **Historical Performance Analysis**: Normalized price tracking and returns distribution
- **Data Persistence**: Automatic CSV export of prices, returns, and allocations
- **Interactive Visualizations**: Plotly-based interactive charts and real-time analysis

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Start the application:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

### Quick Start

1. **Select Mode**:
   - **Manual Selection**: Choose specific assets from predefined categories
   - **AI Selection**: Let the algorithm automatically select 5-10 best assets

2. **Configure Parameters**:
   - Historical Period: 1, 2, 5, or 10 years
   - Capital to Invest: Initial portfolio value
   - Risk-Free Rate: Reference rate for Sharpe ratio calculation
   - Minimum Weight: Minimum allocation per asset

3. **Optimize Portfolio**: Click "Optimize" to generate optimal allocation

4. **Analyze Results**:
   - View portfolio composition and metrics
   - Explore efficient frontier
   - Analyze asset correlations
   - Get capital allocation recommendations
   - Review historical performance
   - Examine risk metrics

## Technical Stack

- **Language**: Python 3.11
- **Web Framework**: Streamlit 1.50+
- **Portfolio Optimization**: PyPortfolioOpt 1.5.5
- **Data Source**: yfinance
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn
- **Optimization Engine**: CVXPY

## Core Components

### 1. Portfolio Optimizer Class

Handles all optimization logic:
- Data fetching from Yahoo Finance
- Returns and covariance calculations
- Efficient frontier computation
- Capital allocation

### 2. AI Asset Selection Engine

Analyzes all available assets:
- Downloads 2-year historical data
- Calculates individual Sharpe ratios
- Ranks assets by risk-adjusted performance
- Returns top N performers

### 3. Interactive Dashboard

Six comprehensive analysis tabs:

**Composition Tab**
- Portfolio weights table
- Allocation pie chart
- CSV/TXT export options

**Efficient Frontier Tab**
- 3000 simulated portfolios
- Color-coded by Sharpe ratio
- Interactive hover information
- Optimal portfolio highlighted

**Correlations Tab**
- Asset correlation heatmap
- Values from -1.0 to +1.0
- Diversification indicators

**Capital Allocation Tab**
- Share quantity recommendations
- Current market prices
- Dollar allocation per asset
- Cash utilization metrics

**Historical Analysis Tab**
- Normalized price performance (Base 100)
- Daily returns distribution histogram
- Multi-year performance comparison

**Risk Metrics Tab**
- Individual asset statistics
- Portfolio VaR and CVaR
- Sharpe ratio analysis
- Risk decomposition

## Modern Portfolio Theory

The application implements Markowitz's mean-variance optimization:

1. **Expected Return Calculation**: Annual returns based on historical data (252 trading days)
2. **Volatility Estimation**: Standard deviation of daily returns, annualized
3. **Correlation Matrix**: Asset co-movement patterns
4. **Optimization Objective**: Maximize (Return - Risk-Free Rate) / Volatility (Sharpe Ratio)
5. **Constraints**:
   - Minimum weight: 5% per asset (configurable)
   - Maximum weight: 50% per asset
   - Full capital investment
   - No short selling

## Available Assets

### Technology
AAPL, MSFT, GOOGL, NVDA, TSLA, META, AMZN, AMD, INTC, CSCO

### Finance
JPM, BAC, WFC, GS, MS, BLK, SCHW, ICE, CME

### Healthcare
JNJ, UNH, PFE, MRK, ABBV, TMO, LLY, BMY

### Energy
XOM, CVX, COP, SLB, MPC, PSX, EOG

### Consumer
WMT, TM, NKE, SBUX, HD, LOW, COST, PG

### Cryptocurrencies
BTC-USD, ETH-USD, SOL-USD, XRP-USD, ADA-USD, DOGE-USD, MATIC-USD, LINK-USD

## Project Structure

```
portfolio-optimizer/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── venv/                 # Virtual environment
├── data/                 # Data storage
│   ├── prices_latest.csv
│   └── returns_latest.csv
└── plots/                # Exported visualizations
    ├── efficient_frontier.html
    ├── price_performance.html
    └── returns_distribution.html
```

## Deployment

### Local Development

```bash
streamlit run app.py
```

### Production Deployment (Streamlit Cloud)

1. Push code to GitHub:
```bash
git add .
git commit -m "Portfolio Optimizer v1.0"
git push origin main
```

2. Deploy on Streamlit Cloud:
   - Visit https://share.streamlit.io
   - Sign in with GitHub account
   - Click "New app"
   - Select repository and branch
   - Set main file to `app.py`
   - Click "Deploy"

3. Share public URL (format: `https://username-portfolio-optimizer-randomid.streamlit.app`)

## Requirements

See `requirements.txt` for complete dependency list:

```
pandas==2.2.3
numpy==2.2.1
yfinance==0.2.66
pyportfolioopt==1.5.5
matplotlib==3.7.1
seaborn==0.13.0
plotly==5.17.0
streamlit==1.50.0
scikit-learn==1.3.2
scipy==1.11.4
```

## API Integration

The application uses:
- **yfinance**: Free Yahoo Finance API for market data
- **CVXPY**: Convex optimization solver for portfolio weights
- **Plotly**: Interactive web-based visualization library

## Performance Considerations

- Data caching: CSV files stored in `data/` directory
- Efficient frontier: 3000 Monte Carlo simulations (configurable)
- Asset analysis: Parallel processing of up to 50+ assets
- Response time: <5 seconds for optimization (typical)

## Risk Disclaimers

This application is provided for educational and informational purposes only. Past performance does not guarantee future results. Portfolio optimization is based on historical data and mathematical models with inherent limitations:

- Historical correlations may not persist
- Market regimes change unpredictably
- Optimization assumes normal return distributions
- Unforeseen events can disrupt market relationships
- Backtested performance may differ from actual results

Always consult with qualified financial advisors before making investment decisions.

## Use Cases

- **Individual Investors**: Construct diversified portfolios with minimum effort
- **Financial Advisors**: Analyze client risk tolerance and asset allocation
- **Quant Researchers**: Test portfolio optimization strategies
- **Academic Study**: Learn Modern Portfolio Theory implementation
- **Portfolio Rebalancing**: Determine optimal weight adjustments

## Future Enhancements

- Machine learning-based return forecasting
- Black-Litterman model for subjective views
- Factor-based asset pricing
- Real-time portfolio monitoring
- Multi-period optimization
- Risk parity portfolios
- Scenario analysis tools

## Contributing

Contributions welcome! Areas for improvement:

- Additional asset classes
- Alternative optimization objectives
- Enhanced visualizations
- Performance optimizations
- Additional risk metrics

Please submit pull requests or issues via GitHub.

## License

MIT License - See LICENSE file for details

## Citation

If you use this application for research, please cite:

```
Portfolio Optimizer: Multi-Asset Portfolio Optimization Platform
Author: David H
Year: 2025
Repository: https://github.com/Sugusdaddy
```

## Contact

For questions, suggestions, or collaboration opportunities:
- Email: vchapixbusiness@gmail.com
- LinkedIn: https://www.linkedin.com/in/david-humada-iglesias-18932a291/

## Acknowledgments

Built with:
- Modern Portfolio Theory (Harry Markowitz)
- CVXPY optimization library
- Streamlit framework
- yfinance data provider

## Disclaimer

This software is provided "as is" without warranty of any kind. Users assume all responsibility for investment decisions. The authors are not liable for any financial losses resulting from the use of this application.

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Status**: Production Ready