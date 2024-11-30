import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class AlwaysPositiveBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """Model that always outputs the positive class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BEGIN YOUR CODE
        # END YOUR CODE
        return None


class AlwaysNegativeBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """Model that always outputs the negative class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BEGIN YOUR CODE
        # END YOUR CODE
        return None
