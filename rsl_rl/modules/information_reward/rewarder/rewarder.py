from torch import Tensor, FloatTensor


class Rewarder:
    """
    A class representing a reward mechanism for given states.
    This class provides a generic interface for inferring rewards based on states and allows for both inferential
    reward computations as well as learning-based reward computations. Methods include functionalities to infer rewards,
    compute explicit rewards, and learn from states.
    """

    def __init__(self, scaling : float = 1.0, concurrent : bool = True) -> None:
        self.scaling = scaling
        self.concurrent = concurrent

    def reward_function(self, states: Tensor) -> FloatTensor:
        """
        Infer rewards for the given states. This method serves as the main mechanism for reward inference
        and can be extended or overridden by specific reward models.
        Args: states (torch.Tensor): States for which the rewards need to be inferred.
        Returns: FloatTensor: Inferred rewards for the provided states.
        """
        raise NotImplementedError()

    def learn(self, states: Tensor) -> None:
        """
        Learn from the provided states. This is an iterative process where the reward mechanism
        adjusts its internal parameters based on the provided states.
        Args: states (torch.Tensor): States from which the reward mechanism will learn.
        Returns: None: This method updates the internal state of the rewarder but does not return any value.
        """
        raise NotImplementedError()
