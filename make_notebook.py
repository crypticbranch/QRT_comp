import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
"""# SCAM MACD Strategy Pipeline
# This notebook implements the end-to-end mathematical pipeline for the Shape-Constrained Additive Model (SCAM) based on the MACD indicator.

import pandas as pd
import numpy as np
import os
import sys
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

# Add scripts folder to path
sys.path.append(os.path.join(os.getcwd(), 'phase2_qrt_challenge', 'scripts'))
import technical_indicators
import utils
""",
"""# Phase 1: Data Infrastructure & Universe Selection
print("Loading data...")
df_historical = pd.read_pickle('top_5000_yf_data.pkl')

if isinstance(df_historical.columns, pd.MultiIndex):
    close_df = df_historical['Close']
    vol_df = df_historical['Volume']
else:
    # If not multi-index, we assume it's just Close prices for now
    close_df = df_historical
    # Fallback if volume is missing: mock volume so ADV logic works (or you should adapt this)
    vol_df = pd.DataFrame(1000000, index=close_df.index, columns=close_df.columns)

# Drop duplicate NAN columns
close_df = close_df.loc[:, ~close_df.columns.duplicated()]
vol_df = vol_df.loc[:, ~vol_df.columns.duplicated()]

print("Data loaded. Shape:", close_df.shape)

# Create 5M ADV Universe
df_daily_volume = close_df.mul(vol_df).fillna(0)
df_adv_60 = df_daily_volume.rolling(window=60, min_periods=1).mean()
universe_df = (df_adv_60 > 5_000_000).astype(int)
print("Universe generated.")
""",
"""# Phase 2: Feature Engineering (The Input)
print("Calculating MACD...")
# technical_indicators.macd returns a tuple: (macd_line, signal_line, histogram)
macd_df, _, _ = technical_indicators.macd(close_df, fast_period=12, slow_period=26)

print("Standardizing MACD...")
# Cross-sectional standardization
macd_mean = macd_df.mean(axis=1).values[:, None]
macd_std = macd_df.std(axis=1).values[:, None]
macd_z = (macd_df - macd_mean) / (macd_std + 1e-8)

# Clip extreme outliers
X_feature = macd_z.clip(-3, 3)
print("Feature engineering complete.")
""",
"""# Phase 3: Target Construction (The Output)
h = 5 # 5-day forward return

print("Calculating returns...")
returns_1d = close_df.pct_change()
returns_5d_fwd = close_df.pct_change(h).shift(-h)

# Market proxy: mean return of the universe
market_1d = returns_1d.mean(axis=1)
market_5d_fwd = returns_5d_fwd.mean(axis=1)

print("Stripping Beta...")
# 60-day rolling beta on 1d returns
rolling_cov = returns_1d.rolling(window=60).cov(market_1d)
rolling_var = market_1d.rolling(window=60).var()
beta = rolling_cov.div(rolling_var, axis=0)

# Calculate idiosyncratic 5-day forward returns
idiosyncratic_5d_fwd = returns_5d_fwd - beta.multiply(market_5d_fwd, axis=0)

print("Standardizing Target...")
target_mean = idiosyncratic_5d_fwd.mean(axis=1).values[:, None]
target_std = idiosyncratic_5d_fwd.std(axis=1).values[:, None]
Y_target = (idiosyncratic_5d_fwd - target_mean) / (target_std + 1e-8)
print("Target construction complete.")
""",
"""# Phase 4: Model Training (The SCAM)
# Flattening matrices to arrays for PyGAM, masking out non-universe and NaNs
mask = (universe_df == 1) & X_feature.notna() & Y_target.notna()

X_flat = X_feature[mask].values.flatten()
Y_flat = Y_target[mask].values.flatten()

# Split train/val by time to implement Embargo
split_idx = int(len(close_df) * 0.7)
split_date = close_df.index[split_idx]
embargo_date = close_df.index[split_idx + h] # 5-day embargo

print(f"Train end: {split_date}, Val start: {embargo_date}")

# Get boolean masks for Train and Validation sets by date index
dates = np.repeat(close_df.index.values[:, None], close_df.shape[1], axis=1)
dates_flat = dates[mask]

train_mask = dates_flat <= split_date.to_numpy()
val_mask = dates_flat >= embargo_date.to_numpy()

X_train, y_train = X_flat[train_mask], Y_flat[train_mask]
X_val, y_val = X_flat[val_mask], Y_flat[val_mask]

# Sample data for speed in training if too large (e.g., take 500k samples)
if len(X_train) > 500000:
    idx = np.random.choice(len(X_train), 500000, replace=False)
    X_train_sub, y_train_sub = X_train[idx], y_train[idx]
else:
    X_train_sub, y_train_sub = X_train, y_train

print(f"Training PyGAM on {len(X_train_sub)} samples...")
# Fit GAM with strictly increasing monotonic constraint
gam = LinearGAM(s(0, constraints='monotonic_inc')).fit(X_train_sub.reshape(-1, 1), y_train_sub)

# Visualize the learned spline
XX = gam.generate_X_grid(term=0)
plt.plot(XX, gam.partial_dependence(term=0, X=XX))
plt.title("Learned Spline for Cross-Sectional MACD")
plt.xlabel("MACD Z-Score")
plt.ylabel("Expected Idiosyncratic Return Z-Score")
plt.grid()
plt.show()
""",
"""# Phase 5: Alpha Scoring & Portfolio Construction
print("Generating predictions for Validation set...")
# We generate predictions for the entire grid to form a portfolio
# But we only construct the portfolio for the validation period
val_dates = close_df.index[close_df.index >= embargo_date]

X_val_matrix = X_feature.loc[val_dates]
universe_val = universe_df.loc[val_dates]
returns_1d_val = returns_1d.loc[val_dates]

# Predict
predictions = pd.DataFrame(index=X_val_matrix.index, columns=X_val_matrix.columns)
# Flatten, predict, reshape
valid_X = X_val_matrix.notna()
preds_flat = gam.predict(X_val_matrix.values[valid_X].reshape(-1, 1))

# Put back into matrix
pred_vals = np.full(X_val_matrix.shape, np.nan)
pred_vals[valid_X] = preds_flat
predictions[:] = pred_vals

print("Volatility Scaling...")
vol_20 = technical_indicators.volatility_20(close_df).loc[val_dates]
# Inverse volatility weighting
alpha_scores = predictions / (vol_20 + 1e-8)

print("Applying constraints (Dollar Neutrality, Universe)...")
# Apply scale_to_book_long_short across rows (each day)
alpha_scaled = alpha_scores.apply(utils.scale_to_book_long_short, axis=1)

# Ensure weights are zero where universe is 0
portfolio = pd.DataFrame(index=alpha_scaled.index, columns=alpha_scaled.columns)
for i in range(len(alpha_scaled)):
    portfolio.iloc[i] = utils.get_universe_adjusted_series(alpha_scaled.iloc[i], universe_val.iloc[i])

portfolio = portfolio.fillna(0)
print("Portfolio construction complete.")
""",
"""# Phase 6: Execution Simulation & Backtest
print("Running Backtest...")
# Utils backtest function simulates turnover and applies a 1bp cost by default
# We can adjust if needed, but we'll use the built-in function
sharpe, gross_pnl = utils.backtest_portfolio(portfolio, returns_1d_val, universe_val, plot_=True, print_=True)
print("Final Net Sharpe Ratio:", sharpe)
"""
]

for cell in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(cell))

with open('scam_macd_backtest.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
