
import yaml

class ExperimentConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_data_config(self):
        return self.config['data']

    def get_model_config(self):
        return self.config['model']

    def get_training_config(self):
        return self.config['training']

    def get_system_config(self):
        return self.config['system']

    def get_seed(self):
        """Get the random seed for reproducibility. Returns None if not set."""
        return self.config.get('system', {}).get('seed', None)
