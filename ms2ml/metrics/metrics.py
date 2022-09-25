from math import pi as PI

import torch
from torch import Tensor
from torch.nn import CosineSimilarity


class CosineLoss(CosineSimilarity):
    """CosineLoss Implements a simple cosine similarity based loss."""

    def __init__(self, dim=1, eps=1e-8) -> None:
        """__init__ Instantiates the class.

        All arguments are passed to `torch.nn.CosineSimilarity`
        """
        super().__init__(dim=dim, eps=eps)

    def forward(self, truth: Tensor, prediction: Tensor) -> Tensor:
        """Forward calculates the loss.

        Parameters:
        -----------
            truth : Tensor
            prediction : Tensor

        Returns:
        --------
            Tensor

        Examples:
        ---------
            >>> loss = CosineLoss(dim=1, eps=1e-4)

            >>> x = torch.ones([1, 2, 5])
            >>> y = torch.zeros([1, 2, 5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals=2)
            tensor([[0., 0., 0., 0., 0.]])
            >>> # Uniform tensors give low loss

            >>> loss(x, 5 * x).round(decimals=2)
            tensor([[0., 0., 0., 0., 0.]])

            >>> loss(torch.zeros([1, 2, 5]), torch.zeros([1, 2, 5]))
            tensor([[1., 1., 1., 1., 1.]])

            >>> loss = CosineLoss(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1], [1, 0.2, 0.1]]]
            >>> x = torch.tensor(x)
            >>> y = [[[0.2, 0.3, 1], [1, 0.2, 0.1]]]
            >>> y = torch.tensor(y)
            >>> x.shape
            torch.Size([1, 2, 3])
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[0.0100, 0.0000]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 13.0]]]
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[0.0000, 0.4900]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[0.000, 0.1000]])
        """
        out = super().forward(truth, prediction)
        out = 1 - out
        return out


class SpectralAngle(torch.nn.CosineSimilarity):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__(dim=dim, eps=eps)

    def forward(self, truth, prediction):
        """Forward calculates the similarity.

        Parameters:
        -----------
            truth : Tensor
            prediction : Tensor

        Returns:
        --------
            Tensor

        Examples:
        ---------
            >>> loss = SpectralAngle(dim=1, eps=1e-4)

            >>> x = torch.ones([1, 2, 5])
            >>> y = torch.zeros([1, 2, 5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals=2)
            tensor([[1., 1., 1., 1., 1.]])

            >>> loss(x, 5 * x).round(decimals=2)
            tensor([[1., 1., 1., 1., 1.]])

            >>> loss(torch.zeros([1, 2, 5]), torch.zeros([1, 2, 5]))
            tensor([[0., 0., 0., 0., 0.]])

            >>> loss = SpectralAngle(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1], [1, 0.2, 0.1]]]
            >>> x = torch.tensor(x)
            >>> y = [[[0.2, 0.3, 1], [1, 0.2, 0.1]]]
            >>> y = torch.tensor(y)
            >>> x.shape
            torch.Size([1, 2, 3])
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[0.9200, nan]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 13.0]]]
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[nan, 0.3400]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[nan, 0.7100]])
        """
        out = super().forward(truth, prediction)
        out = 2 * (torch.acos(out) / PI)
        out = 1 - out

        return out


class SpectralAngleLoss(SpectralAngle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, truth, prediction):
        """Forward calculates the loss.

        Parameters:
        -----------
            truth : Tensor
            prediction : Tensor

        Returns:
        --------
            Tensor

        Examples:
        ---------
            >>> loss = SpectralAngleLoss(dim=1, eps=1e-4)

            >>> x = torch.ones([1, 2, 5])
            >>> y = torch.zeros([1, 2, 5]) + 0.1
            >>> calc_loss = loss(x, y)
            >>> calc_loss.round(decimals=2)
            tensor([[0., 0., 0., 0., 0.]])

            >>> loss(x, 5 * x).round(decimals=2)
            tensor([[0., 0., 0., 0., 0.]])

            >>> loss(torch.zeros([1, 2, 5]), torch.zeros([1, 2, 5]))
            tensor([[1., 1., 1., 1., 1.]])

            >>> loss = SpectralAngleLoss(dim=2, eps=1e-4)
            >>> x = [[[0.1, 0.2, 1], [1, 0.2, 0.1]]]
            >>> x = torch.tensor(x)
            >>> y = [[[0.2, 0.3, 1], [1, 0.2, 0.1]]]
            >>> y = torch.tensor(y)
            >>> x.shape
            torch.Size([1, 2, 3])
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[0.0800, nan]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 13.0]]]
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[nan, 0.6600]])

            >>> x = [[[0.2, 0.4, 2], [1, 0.2, 0.5]]]
            >>> y = [[[0.1, 0.2, 1], [1, 0.2, 0.0]]]
            >>> # The first tensor is a scaled version, and the second
            >>> # has a missmatch
            >>> x = torch.tensor(x)
            >>> y = torch.tensor(y)
            >>> loss(x, y).round(decimals=2).abs()
            tensor([[nan, 0.2900]])
        """
        return 1 - super().forward(truth, prediction)


class PearsonCorrelation(torch.nn.Module):
    """PearsonCorrelation Implements a simple pearson correlation."""

    def __init__(self, axis=1, eps=1e-4):
        """__init__ Instantiates the class.

        Creates a callable object to calculate the pearson correlation on an axis

        Parameters
        ----------
        axis : int, optional
            The axis over which the correlation is calculated.
            For instance, if the input has shape [5, 500] and the axis is set
            to 1, the output will be of shape [5]. On the other hand, if the axis
            is set to 0, the output will have shape [500], by default 1
        eps : float, optional
            Number to be added to to prevent division by 0, by default 1e-4
        """
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x, y):
        """Forward calculates the loss.

        Parameters
        ----------
        truth : Tensor
        prediction : Tensor

        Returns
        -------
        Tensor

        Examples
        --------
        >>> loss = PearsonCorrelation(axis=1, eps=1e-4)
        >>> loss(
        ...     torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ...     torch.tensor([[1.1, 2.0, 3.2], [4.1, 5.0, 6.2]]),
        ... )
        tensor([0.9966, 0.9966])
        >>> out = loss(torch.rand([5, 174]), torch.rand([5, 174]))
        >>> out.shape
        torch.Size([5])
        >>> loss = PearsonCorrelation(axis=0, eps=1e-4)
        >>> out = loss(torch.rand([5, 174]), torch.rand([5, 174]))
        >>> out.shape
        torch.Size([174])
        """
        vx = x - torch.mean(x, axis=self.axis).unsqueeze(self.axis)
        vy = y - torch.mean(y, axis=self.axis).unsqueeze(self.axis)

        num = torch.sum(vx * vy, axis=self.axis)
        denom_1 = torch.sqrt(torch.sum(vx**2, axis=self.axis))
        denom_2 = torch.sqrt(torch.sum(vy**2, axis=self.axis))
        denom = (denom_1 * denom_2) + self.eps
        cost = num / denom
        return cost
