
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import os
from tqdm import tqdm
import torchmetrics

from config import ExperimentConfig
from dataset import WildfireDataset

class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_config = config.get_data_config()
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.system_config = config.get_system_config()

        self._setup_logging()
        self._setup_device()
        self._init_datasets()
        self._init_model()
        self._init_training_components()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("CUDA not available, using CPU")

    def _init_datasets(self):
        self.logger.info("Initializing datasets...")
        self.train_dataset, self.val_dataset, self.test_dataset = WildfireDataset.create_splits(self.config)
        self.logger.debug(f"Train dataset initialized with {len(self.train_dataset)} samples.")
        self.logger.debug(f"Val dataset initialized with {len(self.val_dataset)} samples.")
        self.logger.debug(f"Test dataset initialized with {len(self.test_dataset)} samples.")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.training_config['batch_size'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.training_config['batch_size'])
        
        self.num_features = self.train_dataset.get_num_features()
        self.logger.info(f"Number of features: {self.num_features}")


    def _init_model(self):
        self.logger.info("Initializing model...")
        model_name = self.model_config['name']
        if model_name == 'Transformer':
            params = self.model_config['params']
            self.logger.debug(f"Model parameters: {params}")
            d_model = params['d_model']
            # Input embedding layer to transform from num_features to d_model
            self.input_embedder = nn.Linear(self.num_features, d_model)
            # The output of the transformer will be of shape (batch, seq_len, d_model)
            # We need a linear layer to map this to a single output for binary classification
            self.transformer_model = nn.Transformer(**params)
            self.output_layer = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            # Combine all components into a ModuleList for proper parameter tracking
            self.model = nn.ModuleList([self.input_embedder, self.transformer_model, self.output_layer])
        else:
            self.logger.error(f"Model {model_name} not supported.")
            raise ValueError(f"Model {model_name} not supported.")
        
        # Move model to device
        self.model.to(self.device)
        self.logger.info(f"Model {model_name} initialized and moved to device: {self.device}.")

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
            self.logger.error(f"Optimizer {optimizer_str} not supported.")
            raise ValueError(f"Optimizer {optimizer_str} not supported.")
        self.metrics = self._create_metrics()
        self.best_val_loss = float('inf')
        self.logger.info("Training components initialized.")

    def _create_metrics(self):
        metric_names = self.training_config.get('metrics', [])
        metrics = []
        for name in metric_names:
            if name == "Precision":
                metrics.append(torchmetrics.Precision(task="binary"))
            elif name == "Recall":
                metrics.append(torchmetrics.Recall(task="binary"))
            elif name == "F1Score":
                metrics.append(torchmetrics.F1Score(task="binary"))
            else:
                raise ValueError(f"Metric {name} not supported.")
        return torchmetrics.MetricCollection(metrics).to(self.device)

    def _forward(self, sequences):
        """Forward pass through the model with embedding layer."""
        # Transformer expects (seq_len, batch, features)
        sequences = sequences.permute(1, 0, 2)
        
        # Apply input embedding to transform from num_features to d_model
        embedded = self.input_embedder(sequences)
        
        # For a transformer, source and target can be the same for this task @todo: verify, maybe use wildfire history as target
        transformer_out = self.transformer_model(embedded, embedded)
        
        # Apply output layer
        outputs = self.output_layer(transformer_out)
        
        # Reshape outputs for loss calculation: (seq_len, batch, 1) -> (batch, seq_len)
        outputs = outputs.permute(1, 0, 2).squeeze(-1)
        return outputs

    def train(self):
        self.logger.info("Starting training...")
        max_steps = self.training_config.get('max_steps_per_epoch', None)
        
        for epoch in range(self.training_config['epochs']):
            self.model.train()
            total_steps = len(self.train_loader) if max_steps is None else min(max_steps, len(self.train_loader))
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} / {self.training_config['epochs']} [Train]", leave=True, total=total_steps)
            
            for step, (sequences, labels) in enumerate(train_pbar):
                if max_steps is not None and step >= max_steps:
                    break
                    
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self._forward(sequences)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                self.metrics.update(outputs, labels)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_metrics = self.metrics.compute()
            self.metrics.reset()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {loss.item():.4f}, Metrics: {train_metrics}")

            self._validate(epoch)
        
        self.logger.info("Training finished.")
        self._test()

    def _validate(self, epoch):
        self.model.eval()
        max_steps = self.training_config.get('max_steps_per_epoch', None)
        total_steps = len(self.val_loader) if max_steps is None else min(max_steps, len(self.val_loader))
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} / {self.training_config['epochs']} [Val]", leave=True, total=total_steps)
            for step, (sequences, labels) in enumerate(val_pbar):
                if max_steps is not None and step >= max_steps:
                    break
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self._forward(sequences)
                
                loss = self.loss_fn(outputs, labels)
                total_val_loss += loss.item()
                
                self.metrics.update(outputs, labels)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        val_metrics = self.metrics.compute()
        self.metrics.reset()
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
        self.model.load_state_dict(torch.load(self.system_config['checkpoint_path'], map_location=self.device))
        self.model.eval()
        total_test_loss = 0
        with torch.no_grad():
            test_pbar = tqdm(self.test_loader, desc="[Test]", leave=True)
            for sequences, labels in test_pbar:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self._forward(sequences)

                loss = self.loss_fn(outputs, labels)
                total_test_loss += loss.item()

                self.metrics.update(outputs, labels)
                test_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_test_loss = total_test_loss / len(self.test_loader)
        test_metrics = self.metrics.compute()
        self.metrics.reset()
        self.logger.info(f"Test Loss: {avg_test_loss:.4f}, Metrics: {test_metrics}")
        self.logger.info("Testing finished.")

