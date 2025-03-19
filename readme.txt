Portfolio Optimization with Efficient Frontier
===============================================

Overview:
---------
This project uses historical stock data to perform portfolio optimization based on Modern Portfolio Theory.
It calculates optimal asset allocations using two primary strategies:
  - Maximum Sharpe Ratio Portfolio: Finds the portfolio with the best return per unit of risk.
  - Minimum Variance Portfolio: Finds the portfolio with the lowest possible risk.

It also constructs the efficient frontier – a set of optimal portfolios for varying target returns – and
displays the results using Plotly for interactive visualization.

Features:
---------
• Data Fetching:
  - Uses yfinance to download historical daily closing prices.
  - Calculates daily returns, average returns, and the covariance matrix.

• Portfolio Performance:
  - Computes annualized returns and volatility (risk) based on given weights.
  - Calculates portfolio variance.

• Optimization:
  - Maximizes Sharpe Ratio (by minimizing negative Sharpe Ratio).
  - Minimizes portfolio variance for a safe (low-risk) allocation.
  - Constructs an efficient frontier by optimizing for minimum variance at various target returns.

• Visualization:
  - Uses Plotly to plot:
      – The Maximum Sharpe Ratio portfolio.
      – The Minimum Variance portfolio.
      – The Efficient Frontier (showing optimal risk vs. return trade-offs).

Requirements:
-------------
- Python 3.7+
- numpy
- pandas
- scipy
- yfinance
- plotly

Installation:
-------------
1. Clone the repository:
   git clone https://github.com/mlabib2/portfolio-optimization.git
   cd portfolio-optimization

2. (Optional) Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   (On Windows: venv\Scripts\activate)

3. Install required packages:
   pip install numpy pandas scipy yfinance plotly

Usage:
------
Run the main Python file (e.g., efApp.py) with:
   python efApp.py

The script will:
  • Fetch historical data for a set of example stocks (e.g., CBA.AX, BHP.AX, TLS.AX).
  • Calculate portfolio performance based on user-defined weights.
  • Optimize portfolios to determine the Maximum Sharpe Ratio and Minimum Variance portfolios.
  • Construct and plot the efficient frontier using Plotly.

Code Overview:
--------------
1. Data Fetch (getData):
   - Downloads stock prices, computes daily returns, and calculates mean returns and covariance matrix.

2. Portfolio Performance (portfolioPerformance, portfolioVariance):
   - Computes annualized returns and risk (volatility) for given allocations.

3. Optimization (maxSR, minimizeVariance):
   - Finds optimal weights that maximize the Sharpe Ratio (using negativeSR) and minimize variance.

4. Combined Results (calculatedResults):
   - Aggregates results from both optimizations and builds an efficient frontier.

5. Visualization (EFGraph):
   - Uses Plotly to graph the Maximum Sharpe and Minimum Variance portfolios along with the efficient frontier.

Customization:
--------------
- Modify the stock list in the main section to use different stocks or markets.
- Adjust target returns in the efficientOpt function call to match your investment goals.
- Change the constraint set if you want to allow short-selling or leverage.

Troubleshooting:
----------------
- If the Plotly graph does not display as expected, ensure you are using the correct Plotly version and that your layout is defined with go.Layout (capital L).
- Verify stock symbols or date ranges if data is not returned by yfinance.

License:
--------
This project is licensed under the MIT License.

Acknowledgements:
-----------------
- Data fetched using yfinance.
- Optimization using SciPy.
- Visualizations powered by Plotly.
