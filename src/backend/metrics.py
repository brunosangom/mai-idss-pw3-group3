import torch
import torchmetrics

class MetricsCollection:
    def __init__(self, config):
        self.config = config
        self.metrics = self._create_metrics()
        self.buffer = {'preds': [], 'target': []}

    def _create_metrics(self):
        metric_names = self.config.get_training_config().get('metrics', [])
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
        return torchmetrics.MetricCollection(metrics)

    def store(self, preds, target):
        self.buffer['preds'].append(preds.detach().cpu())
        self.buffer['target'].append(target.detach().cpu())

    def update(self):
        if not self.buffer['preds']:
            return {}
            
        all_preds = torch.cat(self.buffer['preds'])
        all_target = torch.cat(self.buffer['target'])
        
        self.metrics.update(all_preds, all_target)
        computed_metrics = self.metrics.compute()
        
        # Reset for the next epoch/phase
        self.metrics.reset()
        self.buffer = {'preds': [], 'target': []}
        
        return computed_metrics
