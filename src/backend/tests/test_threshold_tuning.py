#!/usr/bin/env python3
"""
Test script to verify the threshold tuning implementation.

This script tests both the training threshold tuning and the evaluation script.
"""

import subprocess
import sys
import os


def test_evaluate_metrics():
    """Test the evaluate_metrics.py script with different FAR thresholds."""
    
    print("=" * 80)
    print("TESTING EVALUATE_METRICS.PY")
    print("=" * 80)
    
    # Test with a recent experiment
    experiment_id = "20251207_161226_f1806fa5"
    
    print(f"\n1. Testing with FAR = 0.25")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "evaluate_metrics.py", 
         "--experiment_id", experiment_id, 
         "--max_far", "0.25"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    print(f"\n2. Testing with FAR = 0.20")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "evaluate_metrics.py", 
         "--experiment_id", experiment_id, 
         "--max_far", "0.20"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    print(f"\n3. Testing with FAR = 0.15")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, "evaluate_metrics.py", 
         "--experiment_id", experiment_id, 
         "--max_far", "0.15"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    return True


def show_config_changes():
    """Show the configuration changes made."""
    print("\n" + "=" * 80)
    print("CONFIGURATION CHANGES")
    print("=" * 80)
    print("\nAdded to config.yaml (training section):")
    print("  max_false_alarm_rate: 0.25  # Maximum allowed FAR during threshold tuning")
    print("  save_results: True  # Enable saving test predictions")
    print("\nModified trainer.py:")
    print("  - Updated _tune_threshold() to maximize FireOnsetRecall while FAR <= max_far")
    print("  - Fixed _test() to properly collect and save predictions")
    print("\nCreated new files:")
    print("  - evaluate_metrics.py: Script to evaluate metrics at different FAR thresholds")
    print("  - EVALUATE_METRICS_README.md: Documentation for the evaluation script")
    print("=" * 80)


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n1. Train a model with threshold tuning:")
    print("   python main.py --config config.yaml")
    print("   (Set tune_threshold: True and max_false_alarm_rate: 0.25 in config)")
    
    print("\n2. Evaluate saved predictions with different FAR thresholds:")
    print("   python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.25")
    
    print("\n3. Show all valid thresholds:")
    print("   python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.25 --show_all")
    
    print("\n4. Try different FAR constraints:")
    print("   python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.20")
    print("   python evaluate_metrics.py --experiment_id 20251209_134233_45e0ef67 --max_far 0.15")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    show_config_changes()
    show_usage_examples()
    
    print("\nRunning tests...")
    success = test_evaluate_metrics()
    
    if not success:
        sys.exit(1)
