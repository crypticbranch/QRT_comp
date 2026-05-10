"""
ROLLING WALK-FORWARD GAM PIPELINE
==================================
Replace Cell 4 (GAM Training) AND Cell 5 (Portfolio Construction) 
with the TWO cells below.

CELL 4 REPLACEMENT: Rolling GAM Training + Prediction
CELL 5 REPLACEMENT: Portfolio Construction from rolling predictions
"""

# ============================================================================
# CELL 4 REPLACEMENT — Rolling Walk-Forward GAM Training
# ============================================================================
# Paste everything below this line into Cell 4:

from pygam import LinearGAM, s, te
import matplotlib.pyplot as plt

# Rolling window config
TRAIN_YEARS = 10        # 10-year training window
FIRST_TRAIN_START = 2010 # First window: 2010-2020 -> trade 2021
LAST_TRADE_YEAR = 2025   # Last year we want to trade

clean_data = stacked_data.dropna()

# Collect predictions across all rolling windows
all_predictions = []

for trade_year in range(FIRST_TRAIN_START + TRAIN_YEARS, LAST_TRADE_YEAR + 1):
    train_start = trade_year - TRAIN_YEARS
    train_end = trade_year - 1
    
    print(f"\n{'='*60}")
    print(f"Window: Train {train_start}-{train_end} -> Trade {trade_year}")
    print(f"{'='*60}")
    
    # --- Split ---
    years = clean_data.index.get_level_values(0).year
    train_mask = (years >= train_start) & (years <= train_end)
    trade_mask = (years == trade_year)
    
    train_data = clean_data[train_mask]
    trade_data = clean_data[trade_mask]
    
    if len(trade_data) == 0:
        print(f"  No trade data for {trade_year}, skipping.")
        continue
    
    X_train = train_data[['volatility', 'rsi', 'momentum', 'rel_volume']].values
    Y_train = train_data['target'].values
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Trading samples:  {len(trade_data):,}")
    
    # --- Fit GAM ---
    gam = LinearGAM(
        s(0, constraints='monotonic_dec', n_splines=10) +  # Volatility
        s(2, constraints='monotonic_inc', n_splines=10) +  # Momentum
        te(1, 3, n_splines=(8, 8))                         # RSI x Rel_Volume
    )
    lam_space = np.logspace(1, 5, 10)
    
    print(f"  Fitting GAM via Gridsearch...")
    gam.gridsearch(X_train, Y_train, lam=lam_space, progress=True)
    
    print(f"  Optimal Lambda: {gam.lam}")
    r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]
    print(f"  Pseudo R-Squared: {r2:.6f}")
    
    # --- Predict on trade year ---
    X_trade = trade_data[['volatility', 'rsi', 'momentum', 'rel_volume']].values
    preds = gam.predict(X_trade)
    
    pred_series = pd.Series(preds, index=trade_data.index, name='prediction')
    all_predictions.append(pred_series)
    
    print(f"  Predictions generated for {trade_year}.")

# Combine all rolling predictions
rolling_predictions = pd.concat(all_predictions)
print(f"\nTotal rolling predictions: {len(rolling_predictions):,}")

# Build the unstacked prediction DataFrame (dates x tickers)
pred_df = rolling_predictions.unstack()

# Build val_data with predictions for portfolio construction
val_data = clean_data.loc[rolling_predictions.index].copy()
val_data['prediction'] = rolling_predictions

print(f"Rolling walk-forward complete. Trade dates: {len(pred_df)}")


# ============================================================================
# CELL 5 REPLACEMENT — Portfolio Construction (unchanged logic)
# ============================================================================
# Paste everything below this line into Cell 5:

import numpy as np
import pandas as pd

# =============================================================================
# Pipeline:
#   1. Predict alpha using the trained GAM (already done above via rolling)
#   2. Apply strict Liquidity Filter (ADV >= $5M)
#   3. Cross-sectionally regress the alpha on betas to extract the residual
#   4. Weight residual signal by inverse realized volatility (with math protections)
#   5. Conviction Threshold with Hysteresis -> Dollar-neutral Long/Short
# =============================================================================

# Parameters
VOL_WINDOW = 20
TARGET_GMV = 10_000_000
MIN_ADV_USD = 5_000_000

# Hysteresis Parameters
ENTRY_THRESHOLD = 2
EXIT_THRESHOLD = 0.5

print('Computing 60-day rolling ADV...')
usd_volume = volume * adj_close
adv_60d = usd_volume.rolling(60, min_periods=30).mean()

print('Computing 20-day realized volatility for signal weighting...')
stock_log_returns = np.log(adj_close / adj_close.shift(1))
realized_vol = stock_log_returns.rolling(VOL_WINDOW).std()
realized_vol_masked = realized_vol.where(universe == 1)

def enforce_limits_iterative(alpha_book, limits_book, target_sum=0.5):
    if alpha_book.sum() == 0:
        return pd.Series(0.0, index=alpha_book.index)
    weights = (alpha_book / alpha_book.sum()) * target_sum
    capped = pd.Series(False, index=weights.index)
    for _ in range(20):
        exceed = weights > limits_book
        if not exceed.any():
            break
        weights[exceed] = limits_book[exceed]
        capped = capped | exceed
        uncapped = ~capped
        if uncapped.sum() == 0:
            break
        remaining_target = target_sum - weights[capped].sum()
        if remaining_target <= 0:
            break
        uncapped_sum = alpha_book[uncapped].sum()
        if uncapped_sum == 0:
            break
        weights[uncapped] = (alpha_book[uncapped] / uncapped_sum) * remaining_target
    return weights

val_dates = pred_df.index
portfolio = pd.DataFrame(0.0, index=val_dates, columns=common_tickers)

print('Building conviction-based portfolio with hysteresis...')
prev_weights = pd.Series(dtype='float64')

for date in val_dates:
    if date not in pred_df.index:
        continue

    alpha = pred_df.loc[date].dropna()
    if len(alpha) < 20:
        continue

    # --- Step 2: STRICT LIQUIDITY FILTER ---
    adv_today = adv_60d.loc[date]
    liquid_mask = adv_today >= MIN_ADV_USD
    alpha = alpha[alpha.index.isin(adv_today[liquid_mask].index)]

    if len(alpha) < 100:
        continue

    # --- Step 3: Regress alpha on betas ---
    beta_today = betas.loc[date, alpha.index].astype('float64')
    valid_mask = beta_today.notna() & alpha.index.isin(beta_today.dropna().index)
    alpha = alpha[valid_mask]
    beta_today = beta_today[alpha.index]

    if len(alpha) < 100:
        continue

    X_beta = beta_today.values
    Y_alpha = alpha.values

    x_mean = np.nanmean(X_beta)
    y_mean = np.nanmean(Y_alpha)
    denom = np.nansum((X_beta - x_mean) ** 2)
    if denom == 0:
        continue

    gamma = np.nansum((X_beta - x_mean) * (Y_alpha - y_mean)) / denom
    intercept = y_mean - gamma * x_mean
    residual = Y_alpha - (gamma * X_beta + intercept)
    residual_signal = pd.Series(residual, index=alpha.index)

    # --- Step 4: Inverse-Vol Weighting ---
    vol_today = realized_vol_masked.loc[date, alpha.index].dropna()
    common_stocks = residual_signal.index.intersection(vol_today.index)

    if len(common_stocks) < 100:
        continue

    residual_signal = residual_signal[common_stocks]
    vol_today = vol_today[common_stocks].clip(lower=0.005)
    vol_weighted_signal = residual_signal / vol_today
    vol_weighted_signal = vol_weighted_signal.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Step 5: Conviction & Hysteresis ---
    z_signal = (vol_weighted_signal - vol_weighted_signal.mean()) / vol_weighted_signal.std()
    z_signal = z_signal.clip(-8.0, 8.0)

    strong_pos = z_signal > ENTRY_THRESHOLD
    strong_neg = z_signal < -ENTRY_THRESHOLD

    if prev_weights.empty:
        hold_pos = pd.Series(False, index=z_signal.index)
        hold_neg = pd.Series(False, index=z_signal.index)
    else:
        prev_aligned = prev_weights.reindex(z_signal.index).fillna(0)
        hold_pos = (z_signal > EXIT_THRESHOLD) & (prev_aligned > 0)
        hold_neg = (z_signal < -EXIT_THRESHOLD) & (prev_aligned < 0)

    alpha_pos = z_signal[strong_pos | hold_pos]
    alpha_neg = z_signal[strong_neg | hold_neg].abs()

    if len(alpha_pos) < 5 or len(alpha_neg) < 5:
        prev_weights = pd.Series(dtype='float64')
        continue

    # --- Step 6: Limits & Allocation ---
    adv_today_filtered = adv_today[z_signal.index]
    max_usd_position = np.minimum(0.025 * adv_today_filtered, 2_000_000)
    max_weight_limit = np.minimum(0.1, max_usd_position / TARGET_GMV)
    limit_pos = max_weight_limit[alpha_pos.index]
    limit_neg = max_weight_limit[alpha_neg.index]

    w_pos = enforce_limits_iterative(alpha_pos, limit_pos, target_sum=0.5)
    w_neg = enforce_limits_iterative(alpha_neg, limit_neg, target_sum=0.5)

    if w_pos.sum() < 0.49 or w_neg.sum() < 0.49:
        prev_weights = pd.Series(dtype='float64')
        continue

    weights = pd.Series(0.0, index=z_signal.index)
    weights[w_pos.index] = w_pos
    weights[w_neg.index] = -w_neg

    portfolio.loc[date, weights.index] = weights.values
    prev_weights = weights.copy()

print(f'Portfolio built. Non-zero days: {(portfolio.abs().sum(axis=1) > 0).sum()}')
