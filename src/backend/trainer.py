
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from config import ExperimentConfig
from dataset import WildfireDataset
from metrics import create_metrics
from models import create_model
from utils import set_seed


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_config = config.get_data_config()
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.system_config = config.get_system_config()
        self.threshold = self.model_config.get('threshold', 0.5)
        self.tune_threshold = self.training_config.get('tune_threshold', False)
        self.system_config['checkpoint_path'] = os.path.join(
            self.system_config.get('checkpoint_dir', 'src/backend/checkpoints/'),
            f"{self.system_config['experiment_id']}.pt"
        )

        self._setup_logging()
        self._setup_seed()
        self._setup_device()
        self._setup_tensorboard()
        self._init_datasets()
        self._init_model()
        self._init_training_components()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def _setup_seed(self):
        """Set random seed for reproducibility if configured."""
        seed = self.config.get_seed()
        if seed is not None:
            set_seed(seed)
            self.logger.info(f"Random seed set to {seed} for reproducibility")
        else:
            self.logger.info("No random seed configured, using default random initialization")

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("CUDA not available, using CPU")

    def _setup_tensorboard(self):
        """Initialize TensorBoard SummaryWriter."""
        log_dir = os.path.join(
            self.system_config.get('log_dir', 'logs/'),
            self.system_config['experiment_id']
        )
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"TensorBoard logs will be saved to {log_dir}")

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
        
        self.model = create_model(
            model_config=self.model_config,
            num_features=self.num_features,
            window_size=self.data_config['window_size'],
            device=self.device
        )
        
        self.logger.info(f"Model {model_name} initialized and moved to device: {self.device}.")

    def _init_training_components(self):
        loss_function_str = self.training_config.get('loss_function', 'BCE')
        self.minority_class_weight = self.training_config.get('minority_class_weight', 1.0)
        self.focal_gamma = self.training_config.get('focal_gamma', 2.0)
        
        if loss_function_str == 'BCE':
            if self.minority_class_weight != 1.0:
                # Use weighted BCE loss to address class imbalance
                # We'll compute the loss manually with per-sample weights
                self.loss_fn = self._weighted_bce_loss
                self.logger.info(f"Using weighted BCE loss with minority_class_weight={self.minority_class_weight}")
            else:
                self.loss_fn = nn.BCELoss()
        elif loss_function_str == 'Focal':
            self.loss_fn = self._focal_loss
            self.logger.info(f"Using Focal Loss with gamma={self.focal_gamma}, alpha={self.minority_class_weight}")
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

    def _weighted_bce_loss(self, outputs, labels):
        """Compute weighted BCE loss to handle class imbalance.
        
        Applies minority_class_weight to positive samples (label=1) and weight 1.0 to negative samples.
        """
        # Create weight tensor: minority_class_weight for positive samples, 1.0 for negative samples
        weights = torch.where(labels == 1, 
                              torch.tensor(self.minority_class_weight, device=labels.device),
                              torch.tensor(1.0, device=labels.device))
        
        # Compute BCE loss per element
        bce = nn.functional.binary_cross_entropy(outputs, labels, reduction='none')
        
        # Apply weights and compute mean
        weighted_bce = bce * weights
        return weighted_bce.mean()

    def _focal_loss(self, outputs, labels):
        """Compute Focal Loss to address class imbalance.
        
        Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Args:
            outputs: Model predictions (probabilities between 0 and 1)
            labels: Ground truth labels (0 or 1)
            
        The focusing parameter gamma reduces the loss contribution from easy examples
        and extends the range in which an example receives low loss.
        - gamma = 0 is equivalent to BCE loss
        - gamma > 0 reduces the relative loss for well-classified examples (p_t > 0.5),
          putting more focus on hard, misclassified examples
        
        Alpha weighting:
        - minority_class_weight is applied to positive samples (label=1)
        - weight 1.0 is applied to negative samples (label=0)
        """
        # Clamp outputs to avoid log(0)
        eps = 1e-7
        outputs = torch.clamp(outputs, eps, 1 - eps)
        
        # Compute p_t (probability of correct class)
        p_t = torch.where(labels == 1, outputs, 1 - outputs)
        
        # Compute alpha_t (class weight for each sample)
        alpha_t = torch.where(labels == 1,
                              torch.tensor(self.minority_class_weight, device=labels.device),
                              torch.tensor(1.0, device=labels.device))
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_loss = -alpha_t * focal_weight * torch.log(p_t)
        
        return focal_loss.mean()

    def _create_metrics(self):
        metric_names = self.training_config.get('metrics', [])
        return create_metrics(metric_names, threshold=self.threshold, device=self.device)

    def _update_metrics(self, preds, targets, prev_targets=None):
        """
        Update all metrics, handling fire onset metrics that require prev_targets.
        
        Args:
            preds: Model predictions for current timestep
            targets: Ground truth labels for current timestep
            prev_targets: Ground truth labels for previous timestep (for onset detection)
        """
        for name, metric in self.metrics.items():
            if 'FireOnset' in name:
                # Fire onset metrics need prev_targets
                metric.update(preds, targets, prev_targets)
            else:
                # Standard metrics
                metric.update(preds, targets)

    def _forward(self, sequences, targets):
        """Forward pass through the model."""
        return self.model(sequences, targets)

    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Threshold used for metrics during training: {self.threshold}")
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
                
                # Prepare targets by shifting labels right and adding a start token (zero)
                zero_tensor = torch.zeros(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
                targets = torch.cat([zero_tensor, labels[:, :-1]], dim=1)
                
                outputs = self._forward(sequences, targets) # (batch, seq_len)

                labels = labels.float()
                # Only compute loss on the last timestep (matches metric evaluation)
                loss = self.loss_fn(outputs[:, -1], labels[:, -1])
                loss.backward()
                self.optimizer.step()
                
                # We only care about the last time step for metrics
                # Pass second-to-last label as prev_target for fire onset detection
                prev_labels = labels[:, -2] if labels.size(1) > 1 else None
                self._update_metrics(outputs[:, -1], labels[:, -1], prev_labels)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_metrics = self.metrics.compute()
            self.metrics.reset()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {loss.item():.4f}, Metrics: {train_metrics}")

            # Log training metrics to TensorBoard
            self.writer.add_scalar('Loss/train', loss.item(), epoch + 1)
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f'{metric_name}/train', metric_value, epoch + 1)

            self._validate(epoch)
        
        self.logger.info("Training finished.")
        
        if self.tune_threshold:
            self._tune_threshold()
        
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
                
                # Prepare targets by shifting labels right and adding a start token (zero)
                zero_tensor = torch.zeros(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
                targets = torch.cat([zero_tensor, labels[:, :-1]], dim=1)
                
                outputs = self._forward(sequences, targets)
                
                labels = labels.float()
                # Only compute loss on the last timestep (matches metric evaluation)
                loss = self.loss_fn(outputs[:, -1], labels[:, -1])
                total_val_loss += loss.item()
                
                # We only care about the last time step for metrics
                # Pass second-to-last label as prev_target for fire onset detection
                prev_labels = labels[:, -2] if labels.size(1) > 1 else None
                self._update_metrics(outputs[:, -1], labels[:, -1], prev_labels)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        val_metrics = self.metrics.compute()
        self.metrics.reset()
        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Metrics: {val_metrics}")

        # Log validation metrics to TensorBoard
        self.writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f'{metric_name}/val', metric_value, epoch + 1)

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_path = self.system_config['checkpoint_path']
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved new best model to {checkpoint_path}")

    def _collect_predictions(self, data_loader, desc="Collecting predictions"):
        """Collect all predictions and labels from a data loader."""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=desc, leave=True)
            for sequences, labels in pbar:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Prepare targets by shifting labels right and adding a start token (zero)
                zero_tensor = torch.zeros(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
                targets = torch.cat([zero_tensor, labels[:, :-1]], dim=1)
                
                outputs = self._forward(sequences, targets)
                
                # We only care about the last time step
                all_probs.append(outputs[:, -1].cpu().numpy())
                all_labels.append(labels[:, -1].cpu().numpy())
        
        return np.concatenate(all_probs), np.concatenate(all_labels)

    def _tune_threshold(self):
        """Tune the classification threshold based on F1 score using validation data."""
        self.logger.info("Tuning threshold based on F1 score...")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.system_config['checkpoint_path'], map_location=self.device))
        
        # Collect predictions on validation set
        y_probs, y_true = self._collect_predictions(self.val_loader, desc="Collecting validation predictions")
        
        # Search for optimal threshold
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            
            # Calculate F1 score
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        self.logger.info(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        
        # Update threshold and recreate metrics with new threshold
        self.threshold = best_threshold
        self.metrics = self._create_metrics()
        self.logger.info(f"Threshold updated to {self.threshold:.4f}")

    def _test(self):
        self.logger.info("Starting testing...")
        self.model.load_state_dict(torch.load(self.system_config['checkpoint_path'], map_location=self.device))
        self.model.eval()
        total_test_loss = 0
        
        # Collect predictions for CSV output
        all_raw_preds = []
        all_pred_wildfires = []
        all_gt_wildfires = []
        
        with torch.no_grad():
            test_pbar = tqdm(self.test_loader, desc="[Test]", leave=True)
            for sequences, labels in test_pbar:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Prepare targets by shifting labels right and adding a start token (zero)
                zero_tensor = torch.zeros(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
                targets = torch.cat([zero_tensor, labels[:, :-1]], dim=1)
                
                outputs = self._forward(sequences, targets)

                labels = labels.float()
                # Only compute loss on the last timestep (matches metric evaluation)
                loss = self.loss_fn(outputs[:, -1], labels[:, -1])
                total_test_loss += loss.item()

                # We only care about the last time step for metrics
                last_step_preds = outputs[:, -1]
                last_step_labels = labels[:, -1]
                prev_labels = labels[:, -2] if labels.size(1) > 1 else None
                self._update_metrics(last_step_preds, last_step_labels, prev_labels)
                
                # Collect predictions for CSV
                all_raw_preds.extend(last_step_preds.cpu().numpy())
                all_pred_wildfires.extend((last_step_preds >= self.threshold).cpu().numpy().astype(int))
                all_gt_wildfires.extend(last_step_labels.cpu().numpy().astype(int))
                
                test_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_test_loss = total_test_loss / len(self.test_loader)
        test_metrics = self.metrics.compute()
        self.metrics.reset()
        self.logger.info(f"Test Loss: {avg_test_loss:.4f}, Metrics: {test_metrics}")
        
        # Log test metrics to TensorBoard
        self.writer.add_scalar('Loss/test', avg_test_loss)
        for metric_name, metric_value in test_metrics.items():
            self.writer.add_scalar(f'{metric_name}/test', metric_value)
        
        # Save test results to CSV if configured
        if self.training_config.get('save_results', True):
            self._save_test_results(all_raw_preds, all_pred_wildfires, all_gt_wildfires)
        
        # Close TensorBoard writer
        self.writer.close()
        
        self.logger.info("Testing finished.")

    def _save_test_results(self, raw_preds, pred_wildfires, gt_wildfires):
        """Save test results to a CSV file with metadata."""        
        # Get metadata from test dataset
        metadata = self.test_dataset.get_all_metadata()
        
        # Verify lengths match
        if len(metadata) != len(raw_preds):
            self.logger.warning(f"Metadata length ({len(metadata)}) doesn't match predictions length ({len(raw_preds)}). "
                              f"This may happen if batch size doesn't evenly divide the dataset.")
            # Truncate to the smaller length
            min_len = min(len(metadata), len(raw_preds))
            metadata = metadata[:min_len]
            raw_preds = raw_preds[:min_len]
            pred_wildfires = pred_wildfires[:min_len]
            gt_wildfires = gt_wildfires[:min_len]
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'latitude': [m[0] for m in metadata],
            'longitude': [m[1] for m in metadata],
            'datetime': [m[2] for m in metadata],
            'raw_pred': raw_preds,
            'pred_wildfire': pred_wildfires,
            'gt_wildfire': gt_wildfires
        })
        
        # Save to CSV in the same directory as logs
        output_path = os.path.join(self.system_config['results_dir'], f'{self.system_config["experiment_id"]}.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Test results saved to {output_path}")

