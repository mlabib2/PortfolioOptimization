import numpy as np
import datetime as dt
import yfinance as yf
import pandas as pd
import scipy as sc

# Function to fetch stock data
def getData(stocks, start, end):
    stock_data = {}
    returns_data = {}

    for stock in stocks:
        try:
            data = yf.download(stock, start=start, end=end)
            if data.empty:
                print(f"No data found for {stock}")
            else:
                stock_data[stock] = data[['Close']]
                returns_data[stock] = stock_data[stock]['Close'].pct_change()  # Compute daily returns
                
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")

    # Convert returns_data dictionary to a DataFrame using `pd.concat()`
    if not returns_data:
        print("No valid data retrieved.")
        return None, None
    
    returns_df = pd.concat(returns_data, axis=1)

    # Compute mean returns and covariance matrix
    meanReturns = returns_df.mean()
    covMatrix = returns_df.cov()

    return meanReturns, covMatrix

# Function to calculate portfolio performance
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252  # Trading days in a year
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun' : lambda x: np.sum(x)-1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    result = sc.optimize.minimize(
        fun=negativeSR, 
        x0=[1./numAssets]*numAssets, 
        args=args,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )

    return result

# Define stock list with Australian market symbols
stocklist = ['CBA', 'BHP', 'TLS']
stocks = [s + '.AX' for s in stocklist]  # Append '.AX' for ASX stocks

# Set date range
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

# Define portfolio weights (converted to NumPy array)
weights = np.array([0.3, 0.3, 0.4])

# Fetch stock data
meanReturns, covMatrix = getData(stocks, startDate, endDate)
if meanReturns is not None:
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix)
    results = maxSR(meanReturns, covMatrix)
    
    # Get the optimal weights from the optimizer result
    opt_weights = results.x
    
    # Calculate the performance of the optimal weights
    opt_returns, opt_std = portfolioPerformance(opt_weights, meanReturns, covMatrix)
    
    print(f"Expected Annual Return: {round(returns * 100, 2)}%")
    print(f"Expected Annual Volatility (Risk): {round(std * 100, 2)}%")
    print(f"Expected Maximum Returns: {round(opt_returns * 100, 2)}%")
else:
    print("No stock return data available.")