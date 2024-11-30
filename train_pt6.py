from typing import Tuple, Optional

from evaluation.eval_structs import EvalMetrics, OperatingPointMetrics
from train import train_and_eval
from trainer.train_structs import TrainMetadata
from utils.utils import create_dataloaders, plot_train_eval_metrics

import torch


def train_and_eval_pt6() -> Tuple[TrainMetadata, Optional[Tuple[EvalMetrics, OperatingPointMetrics]]]:
    """(Optional) Train and evaluate your model for (Part 6).

    Returns:
        (same outputs as: train_and_eval())

    """
    # Feel free to modify everything here, as long as you return the same output types as `train_and_eval()`.
    # You're even welcome to implement a different training loop than the one used in `train_and_eval()`, eg if
    # you want to implement things like: early stopping, etc.
    # BEGIN YOUR CODE
    # END YOUR CODE
    return None, (None, None)


def main():
    train_metadata, (test_eval_metrics, test_metrics_op) = train_and_eval_pt6()
    plot_train_eval_metrics(train_metadata, test_eval_metrics=test_eval_metrics)


if __name__ == '__main__':
    main()
