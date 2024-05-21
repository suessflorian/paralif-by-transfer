import torch

class BernoulliStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        return torch.bernoulli(probs)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient approximation: straight-through estimator
        # Pass the gradient through unchanged
        return grad_output.clone()


def rate_conv(data: torch.Tensor):
    """
    Convert tensor into Poisson spike trains using the features as
    the mean of a binomial distribution.
    Values outside the range of [0, 1] are clipped so they can be
    treated as probabilities.

    Adapted from snntorch spikegen.rate to be differentiable, allowing gradient flow.

        Example::

            # 100% chance of spike generation
            a = torch.Tensor([1, 1, 1, 1])
            spikegen.rate_conv(a)
            >>> tensor([1., 1., 1., 1.])

            # 0% chance of spike generation
            b = torch.Tensor([0, 0, 0, 0])
            spikegen.rate_conv(b)
            >>> tensor([0., 0., 0., 0.])

            # 50% chance of spike generation per time step
            c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
            spikegen.rate_conv(c)
            >>> tensor([0., 1., 0., 1.])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor
    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    # TODO: use torch.nn.functional.softmax instead ?!
    clipped_data = torch.clamp(data, min=0, max=1)
    spike_data = BernoulliStraightThrough.apply(clipped_data)
    return spike_data


def rate(data: torch.Tensor, num_steps=1):
    """
    Generate rate encoding spike trains over a number of steps.

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor
    :param num_steps: Number of time steps to expand the data over.
    :type num_steps: int

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor
    """

    if num_steps <= 0: raise Exception("``num_steps`` must be positive.")
    return rate_conv(data.unsqueeze(0).expand(num_steps, *data.size()))
