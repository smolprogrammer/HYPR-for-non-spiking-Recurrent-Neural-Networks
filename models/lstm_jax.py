import jax
import jax.numpy as jnp
from functools import partial
from models.rnn_cell import InnerRNNCell

# from hypr_trainable_base import GenericHyprTrainableBaseRNNCell
from util.surrogate_derivative import spike_slayer_FGI

import jax.nn.initializers as init
from jax import random

import flax.nnx as nnx
from util.utils import sample_from_range

class LSTM(InnerRNNCell):
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

        self.params = nnx.Param(
            {
                
            }
        )

    @property
    def state_dim(self):
        # cell state, hidden/output, duplicate for recurrent projection
        return 3  # [c, h, o]

    def initialize_carry(self, input_shape):
        batch_size, feature_dim = input_shape
        return jnp.zeros((batch_size, self.num_neurons, self.state_dim))

    def apply_parameter_constraints(self):
        pass

    @property
    def input_multiplier(self) -> int:
        # LSTM has 4 gates
        return 4

    # must be static because of jax.jacfwd
    @staticmethod
    def __call__(input_current, carry, params, hyperparams):
        # i_pre, f_pre, g_pre, o_pre = jnp.split(input_current, 4, axis=-1)
        i_pre, f_pre, g_pre, o_pre =  input_current[..., 0], input_current[..., 1], input_current[..., 2], input_current[..., 3], 


        c_tm1 = carry[..., 0]  # previous cell state
        h_tm1 = carry[..., 1]  # previous hidden/output state

        input_gate = jax.nn.sigmoid(i_pre)
        forget_bias = hyperparams.get("forget_bias", 1.0)
        forget_gate = jax.nn.sigmoid(f_pre + forget_bias)
        candidate = jnp.tanh(g_pre)
        output_gate = jax.nn.sigmoid(o_pre)

        c_t = forget_gate * c_tm1 + input_gate * candidate
        h_t = output_gate * jnp.tanh(c_t)

        # grads of the new output wrt the non-output state components
        d_output_d_c = output_gate * (1.0 - jnp.tanh(c_t) ** 2)
        d_output_d_h = jnp.ones_like(h_t)

        # state, output
        return jnp.stack([c_t, h_t, h_t], axis=-1), (
            h_t,
            jnp.stack([d_output_d_c, d_output_d_h], axis=-1),
        )
