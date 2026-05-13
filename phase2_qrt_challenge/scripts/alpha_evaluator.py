import os
import re
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.alpha_factors import Alpha101


# =============================================================================
# PORTFOLIO BUILDERS
# =============================================================================

def _build_market_neutral_portfolio(signal_df, universe_df):
    """Simple rank-based market-neutral portfolio (existing logic)."""
    tradable = signal_df.shift(1).where(universe_df, np.nan)
    valid_count = tradable.notna().sum(axis=1)
    tradable.loc[valid_count < 20] = np.nan
    ranked = tradable.rank(axis=1)
    weights = ranked.sub(ranked.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1)
    weights = weights.div(abs_sum.replace(0, np.nan), axis=0)
    return weights.fillna(0)


def _enforce_limits_iterative(alpha_book, limits_book, target_sum=0.5):
    """Iteratively cap weights to position limits while maintaining target allocation."""
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
        remaining = target_sum - weights[capped].sum()
        if remaining <= 0:
            break
        usum = alpha_book[uncapped].sum()
        if usum == 0:
            break
        weights[uncapped] = (alpha_book[uncapped] / usum) * remaining
    return weights


def build_conviction_portfolio(
    raw_alpha, returns_df, universe_df, adj_close, volume,
    entry_threshold=2.0, exit_threshold=0.5, min_adv_usd=5_000_000,
    target_gmv=10_000_000, vol_window=20, beta_window=250
):
    """
    Full conviction-based portfolio pipeline:
      1. ADV liquidity filter
      2. Beta-neutralize (cross-sectional regression on rolling betas)
      3. Inverse-vol weighting
      4. Z-score conviction with hysteresis
      5. Position limits (2.5% ADV, $2M max, 10% weight max)
      6. Dollar-neutral long/short (longs=0.5, shorts=-0.5)

    Returns portfolio DataFrame (Date x Ticker).
    """
    # Pre-compute rolling quantities
    usd_volume = volume * adj_close
    adv_60d = usd_volume.rolling(60, min_periods=30).mean()
    log_ret = np.log(adj_close / adj_close.shift(1))
    realized_vol = log_ret.rolling(vol_window).std()

    # Market proxy = equal-weighted mean return
    mkt_ret = returns_df.mean(axis=1)
    # Rolling beta: Beta = 0.2 + 0.8 * cov(stock, mkt) / var(mkt)
    mkt_var = mkt_ret.rolling(beta_window, min_periods=60).var()

    common_dates = raw_alpha.index.intersection(returns_df.index).intersection(universe_df.index)
    common_tickers = (raw_alpha.columns.intersection(returns_df.columns)
                      .intersection(universe_df.columns)
                      .intersection(adj_close.columns))

    raw_alpha = raw_alpha.loc[common_dates, common_tickers]
    rets_aligned = returns_df.loc[common_dates, common_tickers]
    univ_aligned = universe_df.loc[common_dates, common_tickers]

    # Compute rolling betas for all tickers
    print("    Computing rolling betas...")
    betas = pd.DataFrame(index=common_dates, columns=common_tickers, dtype='float64')
    for ticker in common_tickers:
        cov_series = rets_aligned[ticker].rolling(beta_window, min_periods=60).cov(mkt_ret.loc[common_dates])
        betas[ticker] = 0.2 + 0.8 * cov_series / mkt_var.loc[common_dates]

    portfolio = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
    prev_weights = pd.Series(dtype='float64')

    print("    Building conviction portfolio day-by-day...")
    for date in common_dates:
        alpha = raw_alpha.loc[date].dropna()
        if len(alpha) < 20:
            continue

        # Step 1: ADV liquidity filter
        if date in adv_60d.index:
            adv_today = adv_60d.loc[date].reindex(alpha.index)
            liquid = adv_today >= min_adv_usd
            alpha = alpha[liquid & alpha.index.isin(adv_today[liquid].dropna().index)]
        if len(alpha) < 100:
            continue

        # Step 2: Beta-neutralize (regress alpha on beta, take residual)
        beta_today = betas.loc[date, alpha.index].astype('float64')
        valid = beta_today.notna()
        alpha = alpha[valid]
        beta_today = beta_today[alpha.index]
        if len(alpha) < 100:
            continue

        X = beta_today.values
        Y = alpha.values
        x_mean, y_mean = np.nanmean(X), np.nanmean(Y)
        denom = np.nansum((X - x_mean) ** 2)
        if denom == 0:
            continue
        gamma = np.nansum((X - x_mean) * (Y - y_mean)) / denom
        residual = Y - (gamma * X + (y_mean - gamma * x_mean))
        residual_signal = pd.Series(residual, index=alpha.index)

        # Step 3: Inverse-vol weighting
        if date in realized_vol.index:
            vol_today = realized_vol.loc[date].reindex(residual_signal.index).dropna()
        else:
            continue
        common_stocks = residual_signal.index.intersection(vol_today.index)
        if len(common_stocks) < 100:
            continue
        residual_signal = residual_signal[common_stocks]
        vol_today = vol_today[common_stocks].clip(lower=0.005)
        # removed the division by vol_today (Risk Regime Neutrality)
        print("Removed the division by vol_today (Risk Regime Neutrality)")
        vol_weighted = (residual_signal).replace([np.inf, -np.inf], np.nan).dropna()

        # Step 4: Z-score + conviction with hysteresis
        z = (vol_weighted - vol_weighted.mean()) / vol_weighted.std()
        z = z.clip(-8.0, 8.0)

        strong_pos = z > entry_threshold
        strong_neg = z < -entry_threshold

        if prev_weights.empty:
            hold_pos = pd.Series(False, index=z.index)
            hold_neg = pd.Series(False, index=z.index)
        else:
            pa = prev_weights.reindex(z.index).fillna(0)
            hold_pos = (z > exit_threshold) & (pa > 0)
            hold_neg = (z < -exit_threshold) & (pa < 0)

        alpha_pos = z[strong_pos | hold_pos]
        alpha_neg = z[strong_neg | hold_neg].abs()

        if len(alpha_pos) < 5 or len(alpha_neg) < 5:
            prev_weights = pd.Series(dtype='float64')
            continue

        # Step 5: Position limits & allocation
        adv_filt = adv_60d.loc[date].reindex(z.index).fillna(0) if date in adv_60d.index else pd.Series(0, index=z.index)
        max_usd = np.minimum(0.025 * adv_filt, 2_000_000)
        max_wt = np.minimum(0.1, max_usd / target_gmv)

        w_pos = _enforce_limits_iterative(alpha_pos, max_wt[alpha_pos.index], 0.5)
        w_neg = _enforce_limits_iterative(alpha_neg, max_wt[alpha_neg.index], 0.5)

        if w_pos.sum() < 0.49 or w_neg.sum() < 0.49:
            prev_weights = pd.Series(dtype='float64')
            continue

        weights = pd.Series(0.0, index=z.index)
        weights[w_pos.index] = w_pos
        weights[w_neg.index] = -w_neg

        portfolio.loc[date, weights.index] = weights.values
        prev_weights = weights.copy()

    return portfolio


def build_newconviction_portfolio(
    raw_alpha, returns_df, universe_df, adj_close, volume,
    entry_threshold=1.0, exit_threshold=0.25, min_adv_usd=5_000_000,
    target_gmv=10_000_000, vol_window=20, beta_window=250,
    neutralize_beta=False, use_vol_weighting=True, rank_signal=True,
    flag_parabolic=False, flag_market_breadth=False, flag_squeeze_vol=False
):
    """
    New conviction-based portfolio pipeline (modified for Breadth alphas):
      1. ADV liquidity filter
      2. Optional Beta-neutralize
      3. Optional Cross-sectional ranking (crucial for uniform distribution)
      4. Optional Inverse-vol weighting (applied after demeaning)
      5. Z-score conviction with hysteresis (lower thresholds)
      6. Position limits
      7. Dollar-neutral long/short
    """
    usd_volume = volume * adj_close
    adv_60d = usd_volume.rolling(60, min_periods=30).mean()
    log_ret = np.log(adj_close / adj_close.shift(1))
    realized_vol = log_ret.rolling(vol_window).std()

    if neutralize_beta:
        mkt_ret = returns_df.mean(axis=1)
        mkt_var = mkt_ret.rolling(beta_window, min_periods=60).var()

    common_dates = raw_alpha.index.intersection(returns_df.index).intersection(universe_df.index)
    common_tickers = (raw_alpha.columns.intersection(returns_df.columns)
                      .intersection(universe_df.columns)
                      .intersection(adj_close.columns))

    raw_alpha = raw_alpha.loc[common_dates, common_tickers]
    rets_aligned = returns_df.loc[common_dates, common_tickers]
    univ_aligned = universe_df.loc[common_dates, common_tickers]

    if neutralize_beta:
        print("    Computing rolling betas...")
        betas = pd.DataFrame(index=common_dates, columns=common_tickers, dtype='float64')
        for ticker in common_tickers:
            cov_series = rets_aligned[ticker].rolling(beta_window, min_periods=60).cov(mkt_ret.loc[common_dates])
            betas[ticker] = 0.2 + 0.8 * cov_series / mkt_var.loc[common_dates]

    portfolio = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
    prev_weights = pd.Series(dtype='float64')

    if flag_parabolic or flag_market_breadth or flag_squeeze_vol:
        print("    Computing regime filter metrics...")
        sma_20 = adj_close.rolling(20).mean()
        price_to_sma = adj_close / sma_20
        is_parabolic = price_to_sma > 1.30
        
        melt_up_breadth = None
        if flag_market_breadth:
            melt_up_breadth = is_parabolic.mean(axis=1)
            
        is_vol_squeeze = None
        if flag_squeeze_vol:
            realized_vol_20 = log_ret.rolling(20).std()
            historical_vol = log_ret.rolling(250, min_periods=60).std()
            is_vol_squeeze = (realized_vol_20 > 2.0 * historical_vol) & (adj_close > sma_20)

    print("    Building new conviction portfolio day-by-day...")
    for date in common_dates:
        # 1. Apply strict universe mask (crucial for matching Normal portfolio)
        univ_today = univ_aligned.loc[date].fillna(False).astype(bool)
        alpha = raw_alpha.loc[date][univ_today].dropna()

        # Apply Regime Flags
        if flag_parabolic and date in is_parabolic.index:
            parabolic_today = is_parabolic.loc[date].reindex(alpha.index).fillna(False)
            alpha = alpha[~parabolic_today]

        if flag_squeeze_vol and date in is_vol_squeeze.index:
            vol_squeeze_today = is_vol_squeeze.loc[date].reindex(alpha.index).fillna(False)
            alpha = alpha[~vol_squeeze_today]
        
        if len(alpha) < 20:
            continue

        # 2. Apply additional dynamic ADV liquidity filter (optional safety check)
        if date in adv_60d.index:
            adv_today = adv_60d.loc[date].reindex(alpha.index)
            liquid = adv_today >= min_adv_usd
            alpha = alpha[liquid & alpha.index.isin(adv_today[liquid].dropna().index)]
        if len(alpha) < 100:
            continue

        if neutralize_beta:
            beta_today = betas.loc[date, alpha.index].astype('float64')
            valid = beta_today.notna()
            alpha = alpha[valid]
            beta_today = beta_today[alpha.index]
            if len(alpha) < 100:
                continue

            X = beta_today.values
            Y = alpha.values
            x_mean, y_mean = np.nanmean(X), np.nanmean(Y)
            denom = np.nansum((X - x_mean) ** 2)
            if denom == 0:
                continue
            gamma = np.nansum((X - x_mean) * (Y - y_mean)) / denom
            residual = Y - (gamma * X + (y_mean - gamma * x_mean))
            residual_signal = pd.Series(residual, index=alpha.index)
        else:
            residual_signal = alpha
            
        if rank_signal:
            residual_signal = residual_signal.rank()

        # Crucial step: Demean the signal BEFORE applying inverse-volatility weighting!
        # If the signal is strictly positive (e.g. ranks from 1 to N), dividing by vol 
        # without demeaning will completely destroy the short/long symmetry.
        centered_signal = residual_signal - residual_signal.mean()

        if use_vol_weighting:
            if date in realized_vol.index:
                vol_today = realized_vol.loc[date].reindex(centered_signal.index).dropna()
            else:
                continue
            common_stocks = centered_signal.index.intersection(vol_today.index)
            if len(common_stocks) < 100:
                continue
            centered_signal = centered_signal[common_stocks]
            vol_today = vol_today[common_stocks].clip(lower=0.005)
            vol_weighted = (centered_signal / vol_today).replace([np.inf, -np.inf], np.nan).dropna()
        else:
            vol_weighted = centered_signal.replace([np.inf, -np.inf], np.nan).dropna()

        z = (vol_weighted - vol_weighted.mean()) / vol_weighted.std()
        z = z.clip(-8.0, 8.0)

        strong_pos = z > entry_threshold
        strong_neg = z < -entry_threshold

        if prev_weights.empty:
            hold_pos = pd.Series(False, index=z.index)
            hold_neg = pd.Series(False, index=z.index)
        else:
            pa = prev_weights.reindex(z.index).fillna(0)
            hold_pos = (z > exit_threshold) & (pa > 0)
            hold_neg = (z < -exit_threshold) & (pa < 0)

        alpha_pos = z[strong_pos | hold_pos]
        alpha_neg = z[strong_neg | hold_neg].abs()

        if len(alpha_pos) < 5 or len(alpha_neg) < 5:
            prev_weights = pd.Series(dtype='float64')
            continue

        adv_filt = adv_60d.loc[date].reindex(z.index).fillna(0) if date in adv_60d.index else pd.Series(0, index=z.index)
        max_usd = np.minimum(0.025 * adv_filt, 2_000_000)
        max_wt = np.minimum(0.1, max_usd / target_gmv)

        w_pos = _enforce_limits_iterative(alpha_pos, max_wt[alpha_pos.index], 0.5)
        w_neg = _enforce_limits_iterative(alpha_neg, max_wt[alpha_neg.index], 0.5)

        if w_pos.sum() < 0.49 or w_neg.sum() < 0.49:
            prev_weights = pd.Series(dtype='float64')
            continue

        weights = pd.Series(0.0, index=z.index)
        weights[w_pos.index] = w_pos
        weights[w_neg.index] = -w_neg

        if flag_market_breadth and date in melt_up_breadth.index:
            if melt_up_breadth.loc[date] > 0.02:
                weights *= 0.25

        portfolio.loc[date, weights.index] = weights.values
        prev_weights = weights.copy()

    return portfolio


# =============================================================================
# BACKTEST METRICS (QRT-correct costs)
# =============================================================================

def _compute_backtest_metrics(portfolio, returns):
    """
    Transaction costs per QRT Academy User Guide:
      - Execution: 2 bps per unit traded
      - Financing: 0.5% annualised on GMV
    """
    portfolio = portfolio.fillna(0)
    rets = returns.fillna(0)
    gross_pnl = (portfolio * rets).sum(axis=1)
    traded = portfolio.diff(1).abs().sum(axis=1).fillna(0)
    book_value = portfolio.abs().sum(axis=1)
    execution_cost = traded * 2e-4
    financing_cost = book_value * (0.005 / 252)
    net_pnl = gross_pnl - execution_cost - financing_cost

    gs = (gross_pnl.mean() / gross_pnl.std()) * np.sqrt(252) if gross_pnl.std() > 0 else np.nan
    ns = (net_pnl.mean() / net_pnl.std()) * np.sqrt(252) if net_pnl.std() > 0 else np.nan
    to = (traded.mean() / book_value.mean()) * 100 if book_value.mean() > 0 else np.nan
    return {
        "gross_sharpe": round(gs, 3) if pd.notna(gs) else np.nan,
        "net_sharpe": round(ns, 3) if pd.notna(ns) else np.nan,
        "turnover": round(to, 3) if pd.notna(to) else np.nan,
        "gross_pnl": gross_pnl,
    }


# =============================================================================
# YOY EVALUATION
# =============================================================================

def evaluate_conviction_portfolio(
    portfolio, returns_df, raw_alpha, label="Custom"
):
    """Compute YoY metrics for a conviction portfolio and return (metrics, overall)."""
    common_idx = portfolio.index.intersection(returns_df.index)
    common_tk = portfolio.columns.intersection(returns_df.columns)
    port = portfolio.loc[common_idx, common_tk]
    rets = returns_df.loc[common_idx, common_tk]

    # Apply 1-day execution shift
    port = port.shift(1).fillna(0)

    # IC
    if raw_alpha is not None:
        raw_a = raw_alpha.reindex(index=common_idx, columns=common_tk)
        ic_series = raw_a.corrwith(rets.shift(-1), axis=1, method='spearman')
    else:
        ic_series = pd.Series(np.nan, index=common_idx)

    years = sorted(port.index.year.unique())
    metrics = []
    for year in years:
        m = port.index.year == year
        py, ry = port.loc[m], rets.loc[m]
        if py.empty or py.abs().sum().sum() < 1e-10:
            metrics.append({"Year": year, "Net Sharpe": "N/A", "Gross Sharpe": "N/A",
                            "Turnover": "N/A", "IC": "N/A"})
            continue
        bt = _compute_backtest_metrics(py, ry)
        avg_ic = ic_series[ic_series.index.year == year].mean()
        metrics.append({
            "Year": year,
            "Net Sharpe": bt["net_sharpe"] if pd.notna(bt["net_sharpe"]) else "N/A",
            "Gross Sharpe": bt["gross_sharpe"] if pd.notna(bt["gross_sharpe"]) else "N/A",
            "Turnover": f"{bt['turnover']}%" if pd.notna(bt["turnover"]) else "N/A",
            "IC": round(avg_ic, 4) if pd.notna(avg_ic) else "N/A"
        })

    bt_all = _compute_backtest_metrics(port, rets)
    avg_ic_all = ic_series.mean()
    overall = {
        "Net Sharpe": bt_all["net_sharpe"] if pd.notna(bt_all["net_sharpe"]) else "N/A",
        "Gross Sharpe": bt_all["gross_sharpe"] if pd.notna(bt_all["gross_sharpe"]) else "N/A",
        "Turnover": f"{bt_all['turnover']}%" if pd.notna(bt_all["turnover"]) else "N/A",
        "IC": round(avg_ic_all, 4) if pd.notna(avg_ic_all) else "N/A"
    }
    return metrics, overall


# =============================================================================
# ORIGINAL: evaluate_single_alpha (simple rank portfolio)
# =============================================================================

def evaluate_single_alpha(
    alpha_num, yf_data_df, returns_df, universe_df,
    raw_csv_path="raw_alphas.csv", readme_path="alphas_performance.md"
):
    """Compute a single alpha, save raw signal, build simple rank portfolio, report YoY."""
    alpha_name = f"alpha_{alpha_num:03d}"
    print(f"[1/5] Computing {alpha_name}...")

    alphas_obj = Alpha101(yf_data_df, returns_df)
    try:
        raw_signal = getattr(alphas_obj, alpha_name)()
    except AttributeError:
        raise ValueError(f"{alpha_name} does not exist in Alpha101.")
    except Exception as e:
        raise RuntimeError(f"Error computing {alpha_name}: {e}")

    if raw_signal is None or raw_signal.empty:
        print(f"{alpha_name} returned an empty DataFrame.")
        return

    raw_signal = raw_signal.loc[:, ~raw_signal.columns.duplicated()]

    common_index = raw_signal.index.intersection(returns_df.index).intersection(universe_df.index)
    common_tickers = raw_signal.columns.intersection(returns_df.columns).intersection(universe_df.columns)
    raw_aligned = raw_signal.loc[common_index, common_tickers]
    rets = returns_df.loc[common_index, common_tickers]
    univ = universe_df.loc[common_index, common_tickers].fillna(0).astype(bool)

    print(f"    Aligned: {raw_aligned.shape}")

    # Save raw signal
    print(f"[2/5] Saving raw signals to {raw_csv_path}...")
    raw_clean = raw_aligned.dropna(how='all')
    raw_clean.columns = pd.MultiIndex.from_product([[alpha_name], raw_clean.columns], names=['Alpha', 'Ticker'])
    if os.path.exists(raw_csv_path):
        existing = pd.read_csv(raw_csv_path, header=[0, 1], index_col=0, parse_dates=True)
        existing = existing.loc[:, existing.columns.get_level_values(0) != alpha_name]
        if existing.empty:
            raw_clean.to_csv(raw_csv_path)
        else:
            pd.concat([existing, raw_clean], axis=1).to_csv(raw_csv_path)
    else:
        raw_clean.to_csv(raw_csv_path)

    # Build portfolio & metrics
    print(f"[3/5] Generating portfolio for {alpha_name}...")
    portfolio = _build_market_neutral_portfolio(raw_aligned, univ)

    print(f"[4/5] Calculating YoY metrics...")
    ic_series = raw_aligned.corrwith(rets.shift(-1), axis=1, method='spearman')
    years = sorted(portfolio.index.year.unique())
    metrics = []
    for year in years:
        m = portfolio.index.year == year
        py, ry = portfolio.loc[m], rets.loc[m]
        if py.empty or py.abs().sum().sum() < 1e-10:
            metrics.append({"Year": year, "Net Sharpe": "N/A", "Gross Sharpe": "N/A",
                            "Turnover": "N/A", "IC": "N/A"})
            continue
        bt = _compute_backtest_metrics(py, ry)
        avg_ic = ic_series[ic_series.index.year == year].mean()
        metrics.append({
            "Year": year,
            "Net Sharpe": bt["net_sharpe"] if pd.notna(bt["net_sharpe"]) else "N/A",
            "Gross Sharpe": bt["gross_sharpe"] if pd.notna(bt["gross_sharpe"]) else "N/A",
            "Turnover": f"{bt['turnover']}%" if pd.notna(bt["turnover"]) else "N/A",
            "IC": round(avg_ic, 4) if pd.notna(avg_ic) else "N/A"
        })

    bt_all = _compute_backtest_metrics(portfolio, rets)
    avg_ic_all = ic_series.mean()
    overall = {
        "Net Sharpe": bt_all["net_sharpe"] if pd.notna(bt_all["net_sharpe"]) else "N/A",
        "Gross Sharpe": bt_all["gross_sharpe"] if pd.notna(bt_all["gross_sharpe"]) else "N/A",
        "Turnover": f"{bt_all['turnover']}%" if pd.notna(bt_all["turnover"]) else "N/A",
        "IC": round(avg_ic_all, 4) if pd.notna(avg_ic_all) else "N/A"
    }

    print(f"[5/5] Writing results to {readme_path}...")
    _write_readme(readme_path, alpha_name, metrics, overall)
    print(f"✓ Evaluation of {alpha_name} completed.")


def _write_readme(readme_path, alpha_name, metrics, overall):
    """Write or update performance README, replacing existing entry."""
    section_title = alpha_name.replace('_', ' ').title()
    lines = [
        f"\n## {section_title}\n", "",
        f"**Overall** — Net Sharpe: `{overall['Net Sharpe']}` · "
        f"Gross Sharpe: `{overall['Gross Sharpe']}` · "
        f"Turnover: `{overall['Turnover']}` · "
        f"Mean IC: `{overall['IC']}`", "",
        "| Year | Net Sharpe | Gross Sharpe | Turnover | Mean IC |",
        "|------|------------|--------------|----------|---------|",
    ]
    for m in metrics:
        lines.append(f"| {m['Year']} | {m['Net Sharpe']} | {m['Gross Sharpe']} | {m['Turnover']} | {m['IC']} |")
    lines.append("")
    new_section = "\n".join(lines) + "\n"

    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()
        pattern = rf"\n?## {re.escape(section_title)}\n.*?(?=\n## |\Z)"
        content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
        content = content + "\n" + new_section
    else:
        content = new_section
    with open(readme_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    pass
