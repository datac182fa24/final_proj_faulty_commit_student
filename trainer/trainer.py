import torch
from tqdm import trange, tqdm

from evaluation.offline_eval import eval_model
from evaluation.eval_structs import OperatingPointMetrics, EvalMetrics
from trainer.train_structs import TrainMetadata


class ClassificationTrainer:
    """Class that trains a model on a given training dataset with a given criterion/optimizer.
    After each epoch, this runs the offline eval pipeline on the validation dataset.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data_loader: torch.utils.data.DataLoader,
            val_data_loader: torch.utils.data.DataLoader,
            device: torch.device = torch.device("cpu"),
            log_every_n_batches: int = 100,
            skip_val: bool = False,
        ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.log_every_n_batches = log_every_n_batches
        self.skip_val = skip_val

    def perform_validation(self) -> EvalMetrics:
        if self.skip_val:
            print(f"Skipping validation (self.skip_val={self.skip_val})")
            return EvalMetrics(
                torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
                0.0, OperatingPointMetrics(0.0, 0.0, 0.0, 1.0)
            )
        return eval_model(self.model, self.val_data_loader, device=self.device)

    def train(self, total_num_epochs: int) -> TrainMetadata:
        """Trains self.model using given criterion and optimizer.
        Args:
            total_num_epochs: Number of training epochs to do.
        Returns:
            train_metadata:
        """
        self.model = self.model.to(device=self.device)
        self.model.train()
        # BEGIN YOUR CODE
        # END YOUR CODE
        return None
