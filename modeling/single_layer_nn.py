import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class SingleLayerNN(FaultyCommitBinaryClassifierModel):
    """A single layer neural network for binary classification.. The architecture is:
        input -> Linear -> logits
    Equivalent to logistic regression.
    (Test) AP=0.019 (T=0.521) precision@T=0.051 recall@T=0.085 f1@T=0.03187250996015936
    """
    def __init__(self, dim_input_feats: int):
        super().__init__()
        # BEGIN YOUR CODE
        # END YOUR CODE

    def forward(self, x):
        # Tip: this function should return the logits, not class probability (eg don't do Sigmoid here)
        # BEGIN YOUR CODE
        # END YOUR CODE
        return None
