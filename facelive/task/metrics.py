# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Tuple

import torch
from torch import Tensor, tensor 

from torchmetrics.functional.regression.mse import _mean_squared_error_compute, _mean_squared_error_update
from torchmetrics.metric import Metric


from torchmetrics.utilities.checks import _check_same_shape


def _root_mean_squared_error_update(preds: Tensor, target: Tensor, eps:float=1e-6) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Squared Error.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    diff = preds - target
    sum_squared_error = torch.sqrt(torch.sum(diff * diff)+eps)
    n_obs = target.numel()
    return sum_squared_error, n_obs


def _root_mean_squared_error_compute(sum_squared_error: Tensor, n_obs: int, squared: bool = True) -> Tensor:
    """Computes Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_squared_error, n_obs = _mean_squared_error_update(preds, target)
        >>> _mean_squared_error_compute(sum_squared_error, n_obs)
        tensor(0.2500)
    """
    return sum_squared_error / n_obs if squared else torch.sqrt(sum_squared_error / n_obs)

class RootMeanSquaredError(Metric):
    r"""Computes `mean squared error`_ (MSE):

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MeanSquaredError
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mean_squared_error = MeanSquaredError()
        >>> mean_squared_error(preds, target)
        tensor(0.8750)

    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = True,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared
        

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _root_mean_squared_error_update(preds, target, eps=self.eps)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return _root_mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)
