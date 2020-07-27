from catalyst.core.callbacks import MetricCallback
import torch

from catalyst.utils import get_activation_fn

def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = 0.5,
    activation: str = "Sigmoid"
):
    """
    Computes the dice metric
    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        double:  Dice score
    """
    outputs = outputs[:, 0, ...]
    targets = targets[:, 0, ...]

    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice = 2 * (intersection + eps * (union == 0)) / (union + eps)

    return dice

class DiceCallback(MetricCallback):
    """
    Dice metric callback.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our `y_true`.
            output_key (str): output key to use for dice calculation;
                specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )
