
import argparse
from config import ExperimentConfig
from trainer import Trainer

import logging
import os
import uuid
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Wildfire Prediction Experiment")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    system_config = config.get_system_config()
    
    # Setup logging with datetime and unique hash
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    log_dir = system_config.get('log_dir', 'src/backend/logs')
    log_file = os.path.join(log_dir, f'{timestamp}_{unique_id}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    verbose = system_config.get('verbose', False)
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        filename=log_file, 
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded configuration from {args.config}")
    if verbose:
        logger.debug(f"Configuration: {config.config}")

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
