"""
Grid Search for Wildfire Prediction Models

This script performs a grid search over:
- Models: Transformer, LSTM (with varying architecture sizes)
- Loss functions: BCE, Focal
- Minority class weights: configurable

Results are saved to a CSV file for analysis.
"""

import argparse
import itertools
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import torch
import yaml

from config import ExperimentConfig
from trainer import Trainer


# Grid search parameters - Architecture sizes
TRANSFORMER_CONFIGS = [
    # Small
    {
        "name": "Transformer",
        "size": "small",
        "params": {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1
        }
    },
    # Medium
    {
        "name": "Transformer",
        "size": "medium",
        "params": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1
        }
    },
    # Large
    {
        "name": "Transformer",
        "size": "large",
        "params": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.15
        }
    },
]

LSTM_CONFIGS = [
    # Small
    {
        "name": "LSTM",
        "size": "small",
        "params": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False
        }
    },
    # Medium
    {
        "name": "LSTM",
        "size": "medium",
        "params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False
        }
    },
    # Large
    {
        "name": "LSTM",
        "size": "large",
        "params": {
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "bidirectional": False
        }
    },
    # Large Bidirectional
    {
        "name": "LSTM",
        "size": "large_bidir",
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.15,
            "bidirectional": True
        }
    },
]

# Combine all model configs
MODELS = TRANSFORMER_CONFIGS + LSTM_CONFIGS

LOSS_FUNCTIONS = ["BCE", "Focal"]
MINORITY_CLASS_WEIGHTS = [10, 16]  # Focus on higher weights based on class imbalance (~17:1)


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters for grid search."""
    experiments = []
    
    for model, loss_fn, weight in itertools.product(MODELS, LOSS_FUNCTIONS, MINORITY_CLASS_WEIGHTS):
        size_suffix = f"_{model['size']}" if 'size' in model else ""
        experiment = {
            "model": model,
            "loss_function": loss_fn,
            "minority_class_weight": weight,
            "experiment_name": f"{model['name']}{size_suffix}_{loss_fn}_weight{weight}"
        }
        experiments.append(experiment)
    
    return experiments


def create_temp_config(base_config_path: str, experiment: Dict[str, Any], temp_dir: str) -> str:
    """Create a temporary config file with the experiment's hyperparameters."""
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model configuration
    config['model']['name'] = experiment['model']['name']
    config['model']['params'] = experiment['model']['params']
    
    # Update training configuration
    config['training']['loss_function'] = experiment['loss_function']
    config['training']['minority_class_weight'] = experiment['minority_class_weight']
    config['training']['save_results'] = True  # Save results for each experiment
    
    # Create temp config file
    os.makedirs(temp_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_dir, f"config_{experiment['experiment_name']}.yaml")
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config_path


def run_experiment(config_path: str, experiment_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """Run a single experiment and return the results."""
    logger.info(f"Starting experiment: {experiment_name}")
    start_time = time.time()
    
    try:
        config = ExperimentConfig(config_path)
        
        # Set experiment_id in system config (required by Trainer)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        experiment_id = f"{timestamp}_{unique_id}"
        config.config['system']['experiment_id'] = experiment_id
        
        trainer = Trainer(config)
        
        # Train the model (this includes validation and testing)
        trainer.train()
        
        # Get final test metrics from the trainer's last computed metrics
        # Note: The trainer logs metrics but doesn't expose them directly,
        # so we'll re-evaluate on the test set
        test_metrics = evaluate_model(trainer)
        
        elapsed_time = time.time() - start_time
        
        results = {
            "experiment_name": experiment_name,
            "status": "success",
            "elapsed_time": elapsed_time,
            "threshold": trainer.threshold,
            **test_metrics
        }
        
        logger.info(f"Experiment {experiment_name} completed in {elapsed_time:.2f}s")
        logger.info(f"Test metrics: {test_metrics}")
        
    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        logger.error(f"Experiment {experiment_name} failed: {str(e)}")
        logger.error(traceback.format_exc())
        results = {
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "elapsed_time": elapsed_time
        }
    
    return results


def evaluate_model(trainer: Trainer) -> Dict[str, float]:
    """Evaluate the model on the test set and return metrics."""
    trainer.model.eval()
    trainer.metrics.reset()
    
    with torch.no_grad():
        for sequences, labels in trainer.test_loader:
            sequences = sequences.to(trainer.device)
            labels = labels.to(trainer.device)
            
            # Prepare targets by shifting labels right and adding a start token (zero)
            zero_tensor = torch.zeros(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
            targets = torch.cat([zero_tensor, labels[:, :-1]], dim=1)
            
            outputs = trainer.model(sequences, targets)
            
            # We only care about the last time step for metrics
            # Pass second-to-last label as prev_target for fire onset detection
            prev_labels = labels[:, -2].float() if labels.size(1) > 1 else None
            trainer._update_metrics(outputs[:, -1], labels[:, -1].float(), prev_labels)
    
    metrics = trainer.metrics.compute()
    trainer.metrics.reset()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Wildfire Prediction Models")
    parser.add_argument('--config', type=str, default='src/backend/config.yaml',
                        help='Path to the base configuration file.')
    parser.add_argument('--output', type=str, default='src/backend/results/grid_search_results.csv',
                        help='Path to save the grid search results.')
    parser.add_argument('--temp-dir', type=str, default='src/backend/temp_configs',
                        help='Directory for temporary config files.')
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'src/backend/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'grid_search_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Generate experiments
    experiments = generate_experiment_configs()
    logger.info(f"Generated {len(experiments)} experiments for grid search")
    
    # Print experiment summary
    logger.info("=" * 60)
    logger.info("Grid Search Configuration:")
    logger.info(f"  Models: {[m['name'] for m in MODELS]}")
    logger.info(f"  Loss Functions: {LOSS_FUNCTIONS}")
    logger.info(f"  Minority Class Weights: {MINORITY_CLASS_WEIGHTS}")
    logger.info(f"  Total Experiments: {len(experiments)}")
    logger.info("=" * 60)
    
    # Run experiments
    all_results = []
    total_start_time = time.time()
    
    for i, experiment in enumerate(experiments, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment {i}/{len(experiments)}: {experiment['experiment_name']}")
        logger.info(f"  Model: {experiment['model']['name']}")
        logger.info(f"  Loss: {experiment['loss_function']}")
        logger.info(f"  Minority Class Weight: {experiment['minority_class_weight']}")
        logger.info("=" * 60)
        
        # Create temporary config
        temp_config_path = create_temp_config(args.config, experiment, args.temp_dir)
        
        # Run experiment
        results = run_experiment(temp_config_path, experiment['experiment_name'], logger)
        
        # Add experiment parameters to results
        results['model'] = experiment['model']['name']
        results['loss_function'] = experiment['loss_function']
        results['minority_class_weight'] = experiment['minority_class_weight']
        
        all_results.append(results)
        
        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)
        logger.info(f"Intermediate results saved to {args.output}")
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    logger.info("\n" + "=" * 60)
    logger.info("GRID SEARCH COMPLETED")
    logger.info(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)
    
    # Print results summary
    results_df = pd.DataFrame(all_results)
    logger.info("\nResults Summary:")
    logger.info(results_df.to_string())
    
    # Clean up temp configs
    import shutil
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
        logger.info(f"Cleaned up temporary config directory: {args.temp_dir}")


if __name__ == '__main__':
    main()
