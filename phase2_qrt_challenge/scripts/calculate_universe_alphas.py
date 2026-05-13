import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
import multiprocessing as mp
from tqdm import tqdm

# Ensure the parent directory is in the path to allow importing from 'scripts'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.alpha_factors import calculate_all_alphas

# Based on Kakushadze (2016) "101 Formulaic Alphas"
# Alphas 42, 48, 53, and 54 are delay-0 alphas. All others are delay-1 alphas.
# (Note: Alpha 48 requires industry classification and is omitted in our set)
DELAY_0_ALPHAS = ['alpha_042', 'alpha_048', 'alpha_053', 'alpha_054']

def compute_chunk(args):
    """
    Worker function receives pre-sliced data. No disk I/O in the worker.
    """
    yf_chunk, ret_chunk, uni_chunk = args
    
    # Calculate all raw mathematical alphas at time t
    alphas = calculate_all_alphas(yf_chunk, ret_chunk)
    
    if alphas.empty:
        return alphas
        
    # Apply Delays
    for col in alphas.columns:
        alpha_name, ticker = col
        if alpha_name not in DELAY_0_ALPHAS:
            alphas[col] = alphas[col].shift(1)
            
    # Apply universe mask via vectorized MultiIndex alignment
    uni_mask = uni_chunk.replace(0, np.nan)
    alphas = alphas.mul(uni_mask, level=1)
        
    return alphas

def calculate_universe_alphas_parallel(yf_data_path, returns_path, universe_path, output_path, n_jobs=-1, chunk_size=50):
    """
    Calculate alpha factors in parallel, applying delay shifts and universe filters.
    Loads data once into main memory to prevent I/O bottlenecks and OOM errors.
    """
    print("Loading datasets into main memory...")
    with open(yf_data_path, 'rb') as f:
        yf_data = pickle.load(f)
    returns = pd.read_parquet(returns_path)
    universe = pd.read_parquet(universe_path)
    
    all_tickers = universe.columns.tolist()
    
    if n_jobs == -1:
        # Leave one core free so your machine doesn't freeze
        n_jobs = max(1, mp.cpu_count() - 1)
        
    print(f"Slicing data into chunks for {len(all_tickers)} tickers...")
    chunks_data = []
    for i in range(0, len(all_tickers), chunk_size):
        tickers_chunk = all_tickers[i:i + chunk_size]
        # Pre-slice the data so workers don't have to
        chunks_data.append((
            yf_data.loc[:, (slice(None), tickers_chunk)],
            returns[tickers_chunk],
            universe[tickers_chunk]
        ))
    
    print(f"Computing alphas using {n_jobs} workers...")
    results = []
    
    with mp.Pool(n_jobs) as pool:
        for res in tqdm(pool.imap_unordered(compute_chunk, chunks_data), total=len(chunks_data)):
            results.append(res)
            
    print("Concatenating results...")
    final_df = pd.concat(results, axis=1)
    
    # Sort index and columns for MultiIndex neatness
    final_df.sort_index(axis=1, inplace=True)
    
    print(f"Saving to {output_path}...")
    final_df.to_parquet(output_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate 101 Formulaic Alphas for Universe")
    parser.add_argument('--yf-data', type=str, default='../top_5000_yf_data.pkl', help='Path to YF data pickle file')
    parser.add_argument('--returns', type=str, default='../stores/returns.parquet', help='Path to returns parquet file')
    parser.add_argument('--universe', type=str, default='../stores/universe_5m.parquet', help='Path to universe parquet file')
    parser.add_argument('--output', type=str, default='../stores/universe_alpha_factors.parquet', help='Output path')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calculate_universe_alphas_parallel(args.yf_data, args.returns, args.universe, args.output, args.n_jobs)