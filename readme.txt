
1. Daily Returns vs. Prices
	•	What: You have a list of stock closing prices and you convert them into daily returns using pct_change().
	•	Why:
	1.	Returns (percentage changes) are a more direct measure of gains/losses and are needed for portfolio calculations.
	2.	Prices alone don’t tell you how much you actually gained or lost on a day-to-day basis.

⸻

2. Mean Returns & Covariance Matrix
	•	Mean Returns: The average of those daily returns for each asset (e.g., a stock) over a historical period.
	•	Tells you the typical day-to-day gain or loss.
	•	Covariance Matrix: Measures how each pair of stocks moves relative to one another.
	•	Positive Covariance: When Stock A goes up, Stock B tends to go up too.
	•	Negative Covariance: One goes up, the other goes down.
	•	This matters for diversification—assets that move in different ways help reduce overall risk.

Big takeaway: Put these together (mean returns and covariance), and you can approximate how a portfolio might behave (in terms of risk and return).

⸻

3. Annualized Return & Volatility
	1.	Annualized Return:
	•	You see lines like returns * 252 because there are ~252 trading days in a year.
	•	Multiply average daily returns by 252 to estimate a yearly return (e.g., 0.05% a day ~> ~12.6% a year).
	2.	Annualized Volatility (Standard Deviation):
	•	You calculate the daily standard deviation (via the covariance matrix) and then multiply by √252 to scale it up to a yearly figure.
	•	The bigger the standard deviation, the more your returns might swing from day to day (higher “risk”).

⸻

4. Sharpe Ratio
	1.	Formula:
\text{Sharpe Ratio} = \frac{(\text{Portfolio Return} - \text{Risk-Free Rate})}{\text{Volatility}}
	2.	Concept:
	•	Tells you the excess return (above some risk-free investment) per unit of risk (volatility).
	•	A higher Sharpe means you’re getting more “bang” (return) for your “buck” (risk).
	3.	Why “negativeSR”?
	•	Optimization libraries typically minimize. To maximize Sharpe, you minimize -Sharpe.

⸻

5. Maximizing the Sharpe Ratio
	1.	Objective: “Highest risk-adjusted returns.”
	2.	Constraints:
	•	Sum of weights = 1 (fully invested).
	•	Weights between 0 and 1 (no short-selling, no leverage).
	3.	Result: The portfolio that does the best job balancing returns and risk.

⸻

6. Minimizing Variance
	1.	Objective: “Lowest possible volatility,” ignoring how high or low the returns might be.
	2.	Why It Matters:
	•	Some investors prefer stable returns over chasing higher gains.
	•	Minimizing variance is a classic approach to find the least risky portfolio.

⸻

7. Efficient Frontier
	•	A graph showing all optimal portfolios—from the minimum variance one (lowest risk) up to the highest Sharpe one (best risk/reward ratio).
	•	Every point on the frontier is considered efficient because it’s the best you can do for a given level of risk.

⸻

8. Putting It All Together
	1.	Fetch data → daily prices.
	2.	Calculate returns + covariance.
	3.	Run optimization:
	•	Max Sharpe: Find the best ratio of (Return - RF) / Volatility.
	•	Min Variance: Find the smallest standard deviation.
	4.	Interpret the allocations:
	•	If an asset gets a weight of 0%, it suggests that it doesn’t help the particular objective.
	•	If an asset gets a large weight, it suggests it’s either high-return (for Sharpe) or low-risk (for Min Var), or has beneficial correlation patterns.

⸻

9. Practical Insights
	•	No Single “Right” Portfolio:
	•	Max Sharpe is great if you can handle more risk for higher returns.
	•	Min Variance is great if you hate volatility, but might yield lower returns.
	•	Real-World Twists:
	•	Transaction costs, taxes, or constraints on certain assets can change the optimization significantly.
	•	Past performance doesn’t guarantee future returns, so these are estimates based on historical patterns.

⸻

Final Thoughts

These theoretical concepts—mean returns, covariance, Sharpe ratio, volatility, minimization of variance—form the basis of Modern Portfolio Theory. They help you strike a balance between risk (variance/volatility) and return (mean growth), so you can choose a portfolio that aligns with your risk tolerance and investment goals.



