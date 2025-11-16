
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os

from .config import ExperimentConfig
from .dataset import WildfireDataset
from .metrics import MetricsCollection

class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_config = config.get_data_config()
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.system_config = config.get_system_config()

        self._setup_logging()
        self._init_datasets()
        self._init_model()
        self._init_training_components()

    def _setup_logging(self):
        log_file = self.system_config['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _init_datasets(self):
        self.train_dataset = WildfireDataset(self.config, split='train')
        self.val_dataset = WildfireDataset(self.config, split='val')
        self.test_dataset = WildfireDataset(self.config, split='test')

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.training_config['batch_size'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.training_config['batch_size'])
        
        self.num_features = self.train_dataset.get_num_features()


    def _init_model(self):
        model_name = self.model_config['name']
        if model_name == 'Transformer':
            params = self.model_config['params']
            # Adjust d_model to match number of features
            params['d_model'] = self.num_features
            # The output of the transformer will be of shape (batch, seq_len, d_model)
            # We need a linear layer to map this to a single output for binary classification
            transformer_model = nn.Transformer(**params)
            self.model = nn.Sequential(
                transformer_model,
                nn.Linear(self.num_features, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def _init_training_components(self):
        loss_function_str = self.training_config.get('loss_function', 'BCE')
        if loss_function_str == 'BCE':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Loss function {loss_function_str} not supported.")
        optimizer_str = self.training_config.get('optimizer', 'Adam')
        if optimizer_str == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.training_config['learning_rate'])
        else:
            raise ValueError(f"Optimizer {optimizer_str} not supported.")
        self.metrics = MetricsCollection(self.config)
        self.best_val_loss = float('inf')

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.training_config['epochs']):
            self.model.train()
            for sequences, labels in self.train_loader:
                self.optimizer.zero_grad()
                
                # Transformer expects (seq_len, batch, features)
                sequences = sequences.permute(1, 0, 2)
                
                # For a transformer, source and target can be the same for this task @todo: verify, maybe use wildfire history as target
                outputs = self.model(sequences, sequences)
                
                # Reshape outputs and labels for loss calculation
                outputs = outputs.permute(1, 0, 2).squeeze(-1)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                self.metrics.store(outputs, labels)

            train_metrics = self.metrics.update()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {loss.item():.4f}, Metrics: {train_metrics}")

            self._validate(epoch)
        
        self.logger.info("Training finished.")
        self._test()

    def _validate(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.permute(1, 0, 2)
                outputs = self.model(sequences, sequences)
                outputs = outputs.permute(1, 0, 2).squeeze(-1)
                
                loss = self.loss_fn(outputs, labels)
                total_val_loss += loss.item()
                
                self.metrics.store(outputs, labels)
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        val_metrics = self.metrics.update()
        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Metrics: {val_metrics}")

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_path = self.system_config['checkpoint_path']
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved new best model to {checkpoint_path}")

    def _test(self):
        self.logger.info("Starting testing...")
        self.model.load_state_dict(torch.load(self.system_config['checkpoint_path']))
        self.model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.permute(1, 0, 2)
                outputs = self.model(sequences, sequences)
                outputs = outputs.permute(1, 0, 2).squeeze(-1)

                loss = self.loss_fn(outputs, labels)
                total_test_loss += loss.item()

                self.metrics.store(outputs, labels)
        
        avg_test_loss = total_test_loss / len(self.test_loader)
        test_metrics = self.metrics.update()
        self.logger.info(f"Test Loss: {avg_test_loss:.4f}, Metrics: {test_metrics}")
        self.logger.info("Testing finished.")

