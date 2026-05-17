from flax.nnx import RNNCellBase
from typing import Any, Callable, Optional, Tuple

class InnerRNNCell(RNNCellBase):
    def __init__(self, num_neurons, hyperparams, rngs, **kwargs):
        """Base class for RNN cells with parameter constraints."""
        super().__init__(**kwargs)
        self.hyperparams = hyperparams
        self.num_neurons = num_neurons
    
    def apply_parameter_constraints(self) -> None:
        pass

    @property
    def input_multiplier(self) -> int:
        """Number of parallel linear projections required for this cell."""
        return 1
