# Shape-Constrained Additive Model (SCAM) MACD Strategy

This repository contains the quantitative research and implementation for an institutional-grade statistical arbitrage strategy. The strategy leverages a Shape-Constrained Additive Model (SCAM) to extract alpha from the Moving Average Convergence Divergence (MACD) indicator, explicitly controlling for market beta, heteroskedasticity, and overfitting.

## Overview

Traditional technical analysis uses MACD as a binary crossover signal, which is highly susceptible to false positives and curve-fitting. This strategy reimagines MACD as a continuous, cross-sectionally standardized momentum factor. 

By feeding this factor into a Generalized Additive Model (GAM) with strict monotonic constraints, we prevent the model from learning spurious, non-sensical relationships. The model strictly adheres to the financial intuition that higher relative momentum should map to equal or higher expected returns.

## Strategy Assumptions

1. **Market Efficiency:** Markets are mostly efficient, but behavioral biases (herding, slow incorporation of information) create transient momentum anomalies captured by moving averages.
2. **Beta Neutrality vs Dollar Neutrality:** The strategy assumes that sizing the portfolio purely dollar-neutral (Long $1, Short $1) on beta-stripped targets is a sufficient proxy for minimizing systemic market exposure.
3. **Survivorship Bias Acceptance:** For this specific implementation, Yahoo Finance data is used. We accept the known limitation that `yfinance` silently drops delisted tickers, introducing survivorship bias into the backtest.
4. **Monotonicity:** The relationship between cross-sectional MACD and forward idiosyncratic returns is monotonically increasing. Fitting a non-monotonic curve to this relationship is assumed to be curve-fitting to noise.
5. **Liquidity constraint:** Alpha decays rapidly in highly liquid names, but transaction costs destroy alpha in illiquid names. We constrain our universe to the Top 500 stocks by Average Daily Volume (ADV) to balance this tradeoff.

## QRT Competition Constraints Enforced

To comply with the official QRT Academy User Guide limits, the strategy mathematically enforces the following boundaries:
1. **Target Risk ($500k):** The unit-capital portfolio is dynamically scaled such that the ex-ante annualized risk equals exactly $500,000 USD.
2. **Maximum Position Limits:** No individual stock position can exceed $2,000,000 USD or 2.5% of its 60-day Average Daily Volume (ADV).
3. **Turnover & Trading Limits:** Daily rebalancing trades are capped at 2.5% of the stock's 60-day ADV.
4. **Execution assumptions:** 2bps execution cost and 0.5% annualized financing cost on GMV.

## Workflow Pipeline

### Phase 1: Data Infrastructure & Universe Selection
- **Universe:** Top 500 liquid stocks based on a 60-day trailing Average Daily Volume (ADV).
- **Data:** Adjusted Close prices (adjusted for splits and dividends) and Volume.

### Phase 2: Feature Engineering
- **Raw MACD:** $EMA_{12}(Close) - EMA_{26}(Close)$
- **Cross-Sectional Standardization:** At each time step $t$, the MACD is z-scored across the active universe to neutralize market-wide trends.
- **Robustification:** Features are clipped at $\pm 3\sigma$ to prevent outliers from ripping the splines during training.

### Phase 3: Target Construction
- **Forward Returns:** 5-day forward log returns.
- **Beta Stripping:** A 60-day rolling regression is run for each stock against the market (SPY proxy) to isolate idiosyncratic returns.
- **Target Standardization:** The idiosyncratic returns are cross-sectionally z-scored. This mathematically guarantees that minimizing MSE during training directly maximizes the Information Coefficient (IC).

### Phase 4: Model Training (SCAM)
- **Embargoing:** Because overlapping 5-day returns introduce auto-correlation, an embargo of 5 days is enforced between the training set and validation set to prevent data leakage.
- **Model:** A `LinearGAM` from `pygam`.
- **Constraint:** `s(0, constraints='monotonic_inc')` ensures the learned spline strictly increases, regularizing the hypothesis space.

### Phase 5: Alpha Scoring & Portfolio Construction
- **Predictions:** Expected idiosyncratic returns ($\hat{Y}$).
- **Volatility Scaling:** Predictions are divided by trailing 20-day volatility (Inverse Volatility Weighting) to ensure constant risk contribution per asset.
- **Dollar-Neutral Unit Portfolio:** Initial allocations are scaled so the sum of long weights equals $+1$ and short weights equals $-1$.
- **Risk Scaling:** The unit portfolio's daily standard deviation is annualized, and a multiplier is calculated to scale the entire portfolio to a targeted $500,000 Annualized Risk.
- **Constraint Clipping:** The Dollar Portfolio is clipped at $\pm\$2M$, $\pm 2.5\%$ ADV, and the maximum daily trade turnover is capped at $2.5\%$ ADV.

### Phase 6: Execution Simulation
- **Transaction Costs:** 2 bps (0.02%) applied to turnover.
- **Financing Costs:** 0.5% annualized cost applied to Gross Market Value (GMV).
- **Metrics:** Net Sharpe Ratio, Annualized Risk, and Total Return.

## Future Improvements

1. **Survivorship-Bias Free Data:** Migrate from `yfinance` to an institutional provider (e.g., CRSP, Compustat, or Polygon) to include delisted equities.
2. **Explicit Beta Optimization:** Replace naive Dollar-Neutral weighting with a mean-variance optimizer (e.g., CVXPY) that explicitly constrains the portfolio Beta to exactly 0.0.
3. **Turnover Penalization:** MACD is a fast signal. Implement an Exponential Moving Average (EMA) on the final alpha scores or add a turnover penalty in the optimization phase to reduce transaction costs.
4. **Alternative Spline Constraints:** Experiment with concave/convex constraints if diminishing returns are observed at extreme momentum deciles.
5. **Multi-Factor Expansion:** Add a mean-reversion factor (e.g., RSI) as a second spline to capture the interaction between short-term mean reversion and medium-term momentum.


