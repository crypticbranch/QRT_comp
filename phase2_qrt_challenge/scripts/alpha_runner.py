import pandas as pd
import numpy as np
import pickle
import argparse
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

from scripts.alpha_factors import calculate_all_alphas

def compute_chunk(tickers_chunk, yf_data_path, returns_path):
    """
    Worker function to compute alphas for a chunk of tickers.
    Loads data independently to avoid pickling large DataFrames to workers.
    """
    # Load data
    with open(yf_data_path, 'rb') as f:
        yf_data = pickle.load(f)
        
    returns = pd.read_parquet(returns_path)
    
    # Filter for chunk tickers
    yf_chunk = yf_data.loc[:, (slice(None), tickers_chunk)]
    ret_chunk = returns[tickers_chunk]
    
    # Compute
    alphas = calculate_all_alphas(yf_chunk, ret_chunk)
    
    return alphas

def calculate_alphas_parallel(yf_data_path, returns_path, output_path, n_jobs=-1, chunk_size=50):
    """
    Calculate alpha factors in parallel across tickers to handle large datasets.
    """
    # Just read the columns first to get tickers
    returns = pd.read_parquet(returns_path)
    all_tickers = returns.columns.tolist()
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1
        
    # Split tickers into chunks
    chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]
    
    print(f"Computing alphas for {len(all_tickers)} tickers in {len(chunks)} chunks using {n_jobs} workers...")
    
    worker_fn = partial(compute_chunk, yf_data_path=yf_data_path, returns_path=returns_path)
    
    results = []
    with mp.Pool(n_jobs) as pool:
        for res in tqdm(pool.imap_unordered(worker_fn, chunks), total=len(chunks)):
            results.append(res)
            
    print("Concatenating results...")
    final_df = pd.concat(results, axis=1)
    
    # Sort index and columns
    final_df.sort_index(axis=1, inplace=True)
    
    print(f"Saving to {output_path}...")
    final_df.to_parquet(output_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate 101 Formulaic Alphas")
    parser.add_argument('--yf-data', type=str, default='../top_5000_yf_data.pkl', help='Path to YF data pickle file')
    parser.add_argument('--returns', type=str, default='../stores/returns.parquet', help='Path to returns parquet file')
    parser.add_argument('--output', type=str, default='../stores/alpha_factors.parquet', help='Output path')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calculate_alphas_parallel(args.yf_data, args.returns, args.output, args.n_jobs)
