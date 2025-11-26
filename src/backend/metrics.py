import torch
import torchmetrics


class TrueWildfiresMetric(torchmetrics.Metric):
    """Metric that counts the total number of true wildfires (positive labels)."""
    
    def __init__(self):
        super().__init__()
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.count += target.sum().long()
    
    def compute(self):
        return self.count


class PredictedWildfiresMetric(torchmetrics.Metric):
    """Metric that counts the total number of predicted wildfires (predictions above threshold)."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.count += (preds >= self.threshold).sum().long()
    
    def compute(self):
        return self.count


def create_metrics(metric_names: list, threshold: float = 0.5, device: torch.device = None) -> torchmetrics.MetricCollection:
    """
    Create a MetricCollection based on the provided metric names.
    
    Args:
        metric_names: List of metric names to include (e.g., ["Precision", "Recall", "F1Score"])
        threshold: Threshold for binary classification metrics
        device: Device to move the metrics to
    
    Returns:
        A torchmetrics.MetricCollection containing all requested metrics
    """
    metrics = []
    for name in metric_names:
        if name == "Precision":
            metrics.append(torchmetrics.Precision(task="binary", threshold=threshold))
        elif name == "Recall":
            metrics.append(torchmetrics.Recall(task="binary", threshold=threshold))
        elif name == "F1Score":
            metrics.append(torchmetrics.F1Score(task="binary", threshold=threshold))
        elif name == "TrueWildfires":
            metrics.append(TrueWildfiresMetric())
        elif name == "PredictedWildfires":
            metrics.append(PredictedWildfiresMetric(threshold=threshold))
        else:
            raise ValueError(f"Metric {name} not supported.")
    
    collection = torchmetrics.MetricCollection(metrics)
    if device is not None:
        collection = collection.to(device)
    return collection
