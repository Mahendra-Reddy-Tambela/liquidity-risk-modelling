#  Liquidity Risk Modeling in Portfolio Optimization

This project models liquidity risk and incorporates it into portfolio optimization and risk management using Python. It includes real-time data fetching, liquidity metric computation, VaR modeling, and backtesting of liquidity-aware portfolios vs classical Markowitz portfolios.

---

## 📌 Key Highlights

- 📈 **Live Data**: Fetches historical price & volume data for multiple tickers using `yfinance`.
- 🧪 **Liquidity Metrics**: Computes bid-ask spread (simulated), Amihud illiquidity ratio, and turnover ratio.
- 📉 **VaR Analysis**: Compares classical Value at Risk (VaR) with a novel liquidity-adjusted VaR.
- ⚖️ **Portfolio Optimization**: Optimizes portfolio weights using `cvxpy` by penalizing assets with high liquidity costs.
- 🔁 **Backtesting**: Compares classical and liquidity-aware portfolios on cumulative return and max drawdown.

---

## 🛠️ Tools & Libraries

- **Python 3.8+**
- `pandas`, `numpy` – data wrangling
- `yfinance` – financial data
- `scipy.stats` – VaR modeling
- `matplotlib` – visualizations
- `cvxpy` – convex optimization

---