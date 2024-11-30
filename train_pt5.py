from typing import Tuple, Optional

from evaluation.eval_structs import EvalMetrics, OperatingPointMetrics
from train import train_and_eval
from trainer.train_structs import TrainMetadata
from utils.utils import create_dataloaders, plot_train_eval_metrics

import torch


def train_and_eval_pt5() -> Tuple[TrainMetadata, Optional[Tuple[EvalMetrics, OperatingPointMetrics]]]:
    """Train and evaluate your "Improved model" for (Part 5).
    Your goal is to achieve a test AP >= 0.025.

    Returns:
        (same outputs as: train_and_eval())

    """
    # Feel free to modify batchsizes, train_total_num_epochs
    train_batchsize = 1024
    val_batchsize = 1024
    test_batchsize = 1024
    train_total_num_epochs = 10
    numerical_features = [
        'modifications_count',
        'additions_count',
        'deletions_count',
        'hour',
        'day',
        'repo_id',
    ]

    categorical_features = [
        'author_name',
        'author_email',
        'committer_name',
        'committer_email',
        'ext'
    ]

    # Create train/val/test dataloaders
    train_dataloader, val_dataloader, test_dataloader, _ = create_dataloaders(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        batchsize_train=train_batchsize,
        batchsize_val=val_batchsize,
        batchsize_test=test_batchsize,
    )
    dim_input_feats = train_dataloader.dataset[0]["features"].shape[0]
    # Instantiate your improved model (Part 5)
    # BEGIN YOUR CODE
    # END YOUR CODE
    model = None
    return train_and_eval(
        train_batchsize=train_batchsize,
        val_batchsize=val_batchsize,
        test_batchsize=test_batchsize,
        train_total_num_epochs=train_total_num_epochs,
        model=model,
    )


def main():
    train_metadata, (test_eval_metrics, test_metrics_op) = train_and_eval_pt5()
    plot_train_eval_metrics(train_metadata, test_eval_metrics=test_eval_metrics)


if __name__ == '__main__':
    main()
