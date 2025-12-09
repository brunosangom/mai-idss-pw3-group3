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


class FireOnsetMetric(torchmetrics.Metric):
    """
    Metric that measures the model's ability to detect fire onset events.
    
    Fire onset is defined as a transition from no-fire (0) to fire (1).
    This metric tracks:
    - True Positives: Correctly predicted fire onsets
    - False Negatives: Missed fire onsets
    - False Positives: Predicted fire onset when there was no actual onset
    
    The metric computes Precision, Recall, and F1 for fire onset detection.
    
    Note: This metric requires tracking previous labels to detect transitions.
    It uses the `prev_target` parameter in update() to identify onset events.
    If prev_target is not provided, it assumes no previous fire (conservative).
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        # True positives: actual onset AND predicted positive
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        # False negatives: actual onset BUT predicted negative
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        # False positives: predicted positive on non-onset (either no fire or continuing fire)
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        # Total actual onsets
        self.add_state("total_onsets", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, prev_target: torch.Tensor = None):
        """
        Update the metric with new predictions and targets.
        
        Args:
            preds: Model predictions (probabilities)
            target: Current timestep labels (0 or 1)
            prev_target: Previous timestep labels (0 or 1). If None, assumes all 0 (no previous fire).
        """
        binary_preds = (preds >= self.threshold).long()
        target = target.long()
        
        if prev_target is None:
            # If no previous target provided, assume no fire (conservative for onset detection)
            prev_target = torch.zeros_like(target)
        else:
            prev_target = prev_target.long()
        
        # Fire onset: previous was 0, current is 1
        actual_onset = (prev_target == 0) & (target == 1)
        # No onset: either no fire now, or fire was already happening
        no_onset = ~actual_onset
        
        # True positive: actual onset and predicted fire
        self.tp += ((actual_onset) & (binary_preds == 1)).sum()
        # False negative: actual onset but predicted no fire
        self.fn += ((actual_onset) & (binary_preds == 0)).sum()
        # False positive: predicted fire but no actual onset
        # (this includes predicting fire during ongoing fire, which is debatable)
        # For stricter onset detection, we only count FP when target is 0
        self.fp += ((target == 0) & (binary_preds == 1)).sum()
        
        self.total_onsets += actual_onset.sum()
    
    def compute(self):
        """Compute onset detection F1 score."""
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1


class FireOnsetRecall(torchmetrics.Metric):
    """
    Metric that measures recall specifically for fire onset events.
    
    Recall = TP / (TP + FN) = proportion of actual fire onsets that were detected.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, prev_target: torch.Tensor = None):
        binary_preds = (preds >= self.threshold).long()
        target = target.long()
        
        if prev_target is None:
            prev_target = torch.zeros_like(target)
        else:
            prev_target = prev_target.long()
        
        # Fire onset: previous was 0, current is 1
        actual_onset = (prev_target == 0) & (target == 1)
        
        self.tp += ((actual_onset) & (binary_preds == 1)).sum()
        self.fn += ((actual_onset) & (binary_preds == 0)).sum()
    
    def compute(self):
        """Compute onset recall."""
        return self.tp / (self.tp + self.fn + 1e-8)


class FalseAlarmRate(torchmetrics.Metric):
    """
    Metric that measures the False Alarm Rate (False Positive Rate).
    
    False Alarm Rate = FP / (FP + TN) = proportion of actual negatives that were incorrectly predicted as positive.
    
    This is particularly important for wildfire detection systems, as false alarms can lead to:
    - Wasted emergency resources
    - Reduced trust in the system
    - Alert fatigue
    
    Lower values are better (0 is perfect, 1 is worst).
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        binary_preds = (preds >= self.threshold).long()
        target = target.long()
        
        # False positives: predicted fire (1) but actual no fire (0)
        self.fp += ((binary_preds == 1) & (target == 0)).sum()
        # True negatives: predicted no fire (0) and actual no fire (0)
        self.tn += ((binary_preds == 0) & (target == 0)).sum()
    
    def compute(self):
        """Compute False Alarm Rate (FPR)."""
        return self.fp / (self.fp + self.tn + 1e-8)


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
        elif name == "AUROC":
            metrics.append(torchmetrics.AUROC(task="binary"))
        elif name == "TrueWildfires":
            metrics.append(TrueWildfiresMetric())
        elif name == "PredictedWildfires":
            metrics.append(PredictedWildfiresMetric(threshold=threshold))
        elif name == "FireOnsetF1":
            metrics.append(FireOnsetMetric(threshold=threshold))
        elif name == "FireOnsetRecall":
            metrics.append(FireOnsetRecall(threshold=threshold))
        elif name == "FalseAlarmRate":
            metrics.append(FalseAlarmRate(threshold=threshold))
        else:
            raise ValueError(f"Metric {name} not supported.")
    
    # Disable compute_groups to prevent torchmetrics from incorrectly grouping
    # custom metrics that share similar state names (e.g., 'count'), which can
    # cause state corruption between TrueWildfiresMetric and PredictedWildfiresMetric
    collection = torchmetrics.MetricCollection(metrics, compute_groups=False)
    if device is not None:
        collection = collection.to(device)
    return collection
