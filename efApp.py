import numpy as np
import datetime as dt
import yfinance as yf
import pandas as pd
import scipy as sc
from scipy.optimize import minimize


# 1. Data Fetch
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
                returns_data[stock] = stock_data[stock]['Close'].pct_change()
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")

    if not returns_data:
        print("No valid data retrieved.")
        return None, None
    
    returns_df = pd.concat(returns_data, axis=1)
    meanReturns = returns_df.mean()
    covMatrix = returns_df.cov()
    return meanReturns, covMatrix

# 2. Portfolio Performance
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252) #calculates risk volatility
    return returns, std
    # 2. Portfolio Variance
def portfolioVariance(weights, meanReturns, covMatrix):
    _, std = portfolioPerformance(weights, meanReturns, covMatrix)
    return std**2



# 3. Sharpe Ratio Functions
def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = tuple(constraintSet for _ in range(numAssets))
    result = sc.optimize.minimize(
        fun=negativeSR,
        x0=[1./numAssets]*numAssets,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = tuple(constraintSet for _ in range(numAssets))
    result = sc.optimize.minimize(
        fun=portfolioVariance,
        x0=[1./numAssets]*numAssets,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

# 4. Combined Results
def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Return info about the Maximum Sharpe and Minimum Volatility portfolios."""
    # Max Sharpe
    maxSR_Portfolio = maxSR(meanReturns, covMatrix, riskFreeRate, constraintSet) #use NegativeSR() inside maxSR()
    maxSR_Returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix) 
    maxSR_Returns, maxSR_std = round(maxSR_Returns*100,2), round(maxSR_std*100,2)
    maxSR_Allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_Allocation.allocation = [round(i*100,0) for i in maxSR_Allocation.allocation]

    # Min Variance
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix, constraintSet)
    minVol_Returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_Returns, minVol_std = round(minVol_Returns*100,2), round(minVol_std*100,2)
    minVol_Allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_Allocation.allocation = [round(i*100,0) for i in minVol_Allocation.allocation]

    tagetReturns = np.linspace(minVol_Returns, maxSR_Returns,20)
    """This line creates an array of 20 target return values that are evenly spaced between the return of the minimum
      volatility portfolio and the return of the maximum Sharpe ratio portfolio. In other words, if you imagine the
        returns of your safest portfolio (minVol_Returns) and your highest risk-adjusted portfolio (maxSR_Returns) 
        as two endpoints, this function splits that interval into 20 equally spaced points. These targets can then be
          used to construct an efficient frontier by finding the portfolio with minimum risk for each target return."""
    efficientList = []
    for target in tagetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix,target)['fun'])
    return maxSR_Returns, maxSR_std, maxSR_Allocation, minVol_Returns, minVol_std, minVol_Allocation, efficientList


def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0] # zero index to get returns without any volatility 

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet= (0,1)):
    # For each returnTarget, we want optimise the portforio for min variance
    # Given that you want your portfolio to achieve at least a certain return (returnTarget), 
    # find the allocation (weights) of assets that yields the lowest possible risk (variance)
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x,meanReturns,covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    effOpt = minimize(portfolioVariance, numAssets*[1./numAssets,], args=args, method='SLSQP', constraints=constraints, bounds=bounds)
    return effOpt


# 5. Main
if __name__ == "__main__":
    # Example stocks
    stocklist = ['CBA', 'BHP', 'TLS']
    stocks = [s + '.AX' for s in stocklist]

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365)

    # This is optional: a baseline user-chosen weights
    weights = np.array([0.3, 0.3, 0.4])

    meanReturns, covMatrix = getData(stocks, startDate, endDate)
    if meanReturns is not None:
        # Evaluate custom user weights (Optional)
        returns, std = portfolioPerformance(weights, meanReturns, covMatrix)
        print(f"User-Defined Portfolio Return (Annualized): {round(returns*100, 2)}%")
        print(f"User-Defined Portfolio Risk (Annualized):   {round(std*100, 2)}%\n")

        # Print the final summary of Max Sharpe & Min Vol results
        msr_ret, msr_std, msr_alloc, minv_ret, minv_std, minv_alloc, efficientList = calculatedResults(meanReturns, covMatrix)

        print("\n--- Max Sharpe Portfolio ---")
        print(f"Return: {msr_ret}% | Std: {msr_std}%")
        print(msr_alloc)

        print("\n--- Min Vol Portfolio ---")
        print(f"Return: {minv_ret}% | Std: {minv_std}%")
        print(minv_alloc)
        print("_______DIVIDER_____")
        print(efficientOpt(meanReturns, covMatrix, 0.06))
        print("_______DIVIDER_____")
        print(calculatedResults(meanReturns, covMatrix))

    else:
        print("No stock return data available.")

        