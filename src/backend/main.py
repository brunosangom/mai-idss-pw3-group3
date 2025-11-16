
import argparse
from config import ExperimentConfig
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Wildfire Prediction Experiment")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
