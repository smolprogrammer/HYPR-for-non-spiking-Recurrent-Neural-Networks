import jax
import jax.numpy as jnp
import flax.nnx as nnx

from models.rnn_cell import InnerRNNCell


class GRU(InnerRNNCell):
    def __init__(
        self,
        num_neurons: int,
        hyperparams,
        rngs,
        **kwargs,
    ):
        super().__init__(
            num_neurons,
            hyperparams,
            rngs,
            **kwargs,
        )

        # GRU has no per-neuron parameters; recurrent weights live in the wrapper layer
        self.params = nnx.Param({})

    @property
    def state_dim(self):
        # no cell state
        return 2 # [h, o]

    def initialize_carry(self, input_shape):
        batch_size, _ = input_shape
        return jnp.zeros((batch_size, self.num_neurons, self.state_dim))

    def apply_parameter_constraints(self):
        pass

    @property
    def input_multiplier(self) -> int:
        # three gates: update, reset, candidate
        return 3

    @staticmethod
    def __call__(input_current, carry, params, hyperparams):
        # input_current expected shape (..., 3) ordered as [update, reset, candidate]
        if __debug__:
            assert input_current.shape[-1] == 3
        z_pre = input_current[..., 0]
        r_pre = input_current[..., 1]
        n_pre = input_current[..., 2]

        h_tm1 = carry[..., 0]

        update_gate = jax.nn.sigmoid(z_pre)
        reset_gate = jax.nn.sigmoid(r_pre)

        # Candidate uses only per-neuron projected preactivations; no cross-neuron mixing here
        candidate = jnp.tanh(reset_gate * n_pre)

        h_t = (1.0 - update_gate) * candidate + update_gate * h_tm1
        o_t = h_t

        d_output_d_h = jnp.ones_like(h_t)

        return jnp.stack([h_t, o_t], axis=-1), (
            o_t,
            d_output_d_h[..., None],
        ) 
