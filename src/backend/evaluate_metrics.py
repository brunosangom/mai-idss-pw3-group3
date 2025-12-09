#!/usr/bin/env python3
"""
Script to evaluate metrics for a given experiment ID and False Alarm Rate threshold.

This script reads saved test predictions from the results directory and computes
metrics for different classification thresholds that satisfy a given FAR constraint.

Usage:
    python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.25
    python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.20 --show_all
"""

import argparse
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    Calculate all relevant metrics for wildfire prediction.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Binary predictions (0 or 1)
        y_probs: Prediction probabilities [0, 1]
    
    Returns:
        Dictionary of metric names and values
    """
    # Basic classification metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # False Alarm Rate
    far = fp / (fp + tn + 1e-8)
    
    # AUROC using sklearn if available
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_probs)
    except ImportError:
        auroc = None
    
    # Count metrics
    true_wildfires = np.sum(y_true == 1)
    pred_wildfires = np.sum(y_pred == 1)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1Score': f1,
        'FalseAlarmRate': far,
        'TrueWildfires': int(true_wildfires),
        'PredictedWildfires': int(pred_wildfires),
        'TruePositives': int(tp),
        'TrueNegatives': int(tn),
        'FalsePositives': int(fp),
        'FalseNegatives': int(fn),
    }
    
    if auroc is not None:
        metrics['AUROC'] = auroc
    
    return metrics


def calculate_fire_onset_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  metadata_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fire onset detection metrics.
    
    Fire onset is defined as a transition from no-fire to fire at the same location.
    We need to identify previous timestep labels for each sample.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Binary predictions (0 or 1)
        metadata_df: DataFrame with latitude, longitude, datetime columns
    
    Returns:
        Dictionary of fire onset metrics
    """
    # Sort by location and time
    df = metadata_df.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['latitude', 'longitude', 'datetime'])
    
    # Calculate previous label for each location
    df['prev_true'] = df.groupby(['latitude', 'longitude'])['y_true'].shift(1, fill_value=0)
    
    # Fire onset: previous was 0, current is 1
    actual_onset = (df['prev_true'] == 0) & (df['y_true'] == 1)
    
    # Calculate onset metrics
    tp_onset = np.sum((actual_onset) & (df['y_pred'] == 1))
    fn_onset = np.sum((actual_onset) & (df['y_pred'] == 0))
    
    fire_onset_recall = tp_onset / (tp_onset + fn_onset + 1e-8)
    total_onsets = np.sum(actual_onset)
    
    return {
        'FireOnsetRecall': fire_onset_recall,
        'TotalOnsets': int(total_onsets),
        'DetectedOnsets': int(tp_onset),
        'MissedOnsets': int(fn_onset),
    }


def find_optimal_threshold(results_df: pd.DataFrame, max_far: float = 0.25, 
                           num_thresholds: int = 99) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal classification threshold that maximizes FireOnsetRecall
    while keeping False Alarm Rate below max_far.
    
    Args:
        results_df: DataFrame with columns: raw_pred, gt_wildfire, latitude, longitude, datetime
        max_far: Maximum allowed False Alarm Rate
        num_thresholds: Number of threshold values to try
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    y_probs = results_df['raw_pred'].values
    y_true = results_df['gt_wildfire'].values
    
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_recall = -1
    best_threshold = 0.5
    best_metrics = None
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calculate FAR
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        far = fp / (fp + tn + 1e-8)
        
        # Only consider thresholds that satisfy FAR constraint
        if far > max_far:
            continue
        
        # Calculate all metrics
        basic_metrics = calculate_metrics(y_true, y_pred, y_probs)
        onset_metrics = calculate_fire_onset_metrics(y_true, y_pred, results_df)
        
        metrics = {**basic_metrics, **onset_metrics, 'Threshold': thresh}
        
        # Select threshold with highest FireOnsetRecall
        if onset_metrics['FireOnsetRecall'] > best_recall:
            best_recall = onset_metrics['FireOnsetRecall']
            best_threshold = thresh
            best_metrics = metrics
    
    return best_threshold, best_metrics


def evaluate_at_threshold(results_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """
    Evaluate metrics at a specific threshold.
    
    Args:
        results_df: DataFrame with columns: raw_pred, gt_wildfire, latitude, longitude, datetime
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_probs = results_df['raw_pred'].values
    y_true = results_df['gt_wildfire'].values
    y_pred = (y_probs >= threshold).astype(int)
    
    basic_metrics = calculate_metrics(y_true, y_pred, y_probs)
    onset_metrics = calculate_fire_onset_metrics(y_true, y_pred, results_df)
    
    return {**basic_metrics, **onset_metrics, 'Threshold': threshold}


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate metrics for saved test predictions at different False Alarm Rate thresholds'
    )
    parser.add_argument(
        '--experiment_id', 
        type=str, 
        required=True,
        help='Experiment ID (e.g., 20251209_134233_45e0ef67)'
    )
    parser.add_argument(
        '--max_far', 
        type=float, 
        default=0.25,
        help='Maximum False Alarm Rate threshold (default: 0.25)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/',
        help='Directory containing results CSV files (default: results/)'
    )
    parser.add_argument(
        '--show_all',
        action='store_true',
        help='Show all thresholds that satisfy FAR constraint (not just optimal)'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_path = os.path.join(args.results_dir, f'{args.experiment_id}.csv')
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print(f"Make sure save_results=True in config and the experiment has been run.")
        return
    
    print(f"Loading results from {results_path}...")
    results_df = pd.read_csv(results_path)
    
    # Validate required columns
    required_cols = ['raw_pred', 'gt_wildfire', 'latitude', 'longitude', 'datetime']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    print(f"Loaded {len(results_df)} test predictions")
    print(f"Total wildfires: {results_df['gt_wildfire'].sum()}")
    print()
    
    # Find optimal threshold
    print(f"Finding optimal threshold with max FAR = {args.max_far:.4f}...")
    optimal_threshold, optimal_metrics = find_optimal_threshold(results_df, args.max_far)
    
    if optimal_metrics is None:
        print(f"Error: No threshold found that satisfies FAR <= {args.max_far:.4f}")
        print("Try increasing max_far value.")
        return
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLD: {optimal_threshold:.4f}")
    print(f"{'='*70}")
    print(f"FireOnsetRecall:     {optimal_metrics['FireOnsetRecall']:.4f}")
    print(f"FalseAlarmRate:      {optimal_metrics['FalseAlarmRate']:.4f}")
    print(f"Precision:           {optimal_metrics['Precision']:.4f}")
    print(f"Recall:              {optimal_metrics['Recall']:.4f}")
    print(f"F1Score:             {optimal_metrics['F1Score']:.4f}")
    if 'AUROC' in optimal_metrics:
        print(f"AUROC:               {optimal_metrics['AUROC']:.4f}")
    print()
    print(f"Total Onsets:        {optimal_metrics['TotalOnsets']}")
    print(f"Detected Onsets:     {optimal_metrics['DetectedOnsets']}")
    print(f"Missed Onsets:       {optimal_metrics['MissedOnsets']}")
    print()
    print(f"True Wildfires:      {optimal_metrics['TrueWildfires']}")
    print(f"Predicted Wildfires: {optimal_metrics['PredictedWildfires']}")
    print(f"True Positives:      {optimal_metrics['TruePositives']}")
    print(f"True Negatives:      {optimal_metrics['TrueNegatives']}")
    print(f"False Positives:     {optimal_metrics['FalsePositives']}")
    print(f"False Negatives:     {optimal_metrics['FalseNegatives']}")
    print(f"{'='*70}")
    
    # Show all valid thresholds if requested
    if args.show_all:
        print(f"\n{'='*70}")
        print(f"ALL THRESHOLDS WITH FAR <= {args.max_far:.4f}")
        print(f"{'='*70}")
        
        thresholds = np.linspace(0.01, 0.99, 99)
        valid_thresholds = []
        
        for thresh in thresholds:
            y_probs = results_df['raw_pred'].values
            y_true = results_df['gt_wildfire'].values
            y_pred = (y_probs >= thresh).astype(int)
            
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            far = fp / (fp + tn + 1e-8)
            
            if far <= args.max_far:
                metrics = evaluate_at_threshold(results_df, thresh)
                valid_thresholds.append(metrics)
        
        if not valid_thresholds:
            print("No valid thresholds found.")
        else:
            # Sort by FireOnsetRecall descending
            valid_thresholds.sort(key=lambda x: x['FireOnsetRecall'], reverse=True)
            
            print(f"\n{'Threshold':>10} {'FAR':>10} {'OnsetRec':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print('-' * 70)
            for m in valid_thresholds[:20]:  # Show top 20
                print(f"{m['Threshold']:>10.4f} {m['FalseAlarmRate']:>10.4f} "
                      f"{m['FireOnsetRecall']:>10.4f} {m['Precision']:>10.4f} "
                      f"{m['Recall']:>10.4f} {m['F1Score']:>10.4f}")
            
            if len(valid_thresholds) > 20:
                print(f"... ({len(valid_thresholds) - 20} more thresholds)")


if __name__ == '__main__':
    main()
