import torch

from modeling.model_interface import FaultyCommitBinaryClassifierModel


class RandomBinaryClassifier(FaultyCommitBinaryClassifierModel):
    """
    prob_predict_positive=0.5:
    (Test) AP=0.013 (T=0.000) precision@T=0.014 recall@T=1.000 f1@T=0.01345007755735162
    """
    def __init__(self, prob_predict_positive: float = 0.5):
        """

        Args:
            prob_predict_positive:
                Value between [0.0, 1.0].
                0.0: never predict positive
                1.0: always predict positive
        """
        super().__init__()
        self.prob_predict_positive = prob_predict_positive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: Input features.
                shape=[batchsize, dim_feats].
                For details on format, see: `FaultCSVDataset`.

        Returns:
            is_faulty_commit_logit:
                shape=[batchsize, 1]

        """
        # BEGIN YOUR CODE
        # END YOUR CODE
        return None
