from typing import Type
from flax.nnx.nn.recurrent import RNNCellBase
from models.rnn_cell import InnerRNNCell
import flax.nnx as nnx
import jax

from util.utils import get_initializer


class RNNLayerWithDense(RNNCellBase):
    def __init__(
        self,
        cell_class: Type[InnerRNNCell],
        cell_hyperparams: dict,
        in_features: int,
        out_features: int,
        has_recurrent_connections: bool,
        kernel_initializer,
        rec_initializer,
        use_bias,
        rngs,
    ):
        self.cell = cell_class(
            num_neurons=out_features,
            hyperparams=cell_hyperparams,
            rngs=rngs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.has_recurrent_connections = has_recurrent_connections
        proj_multiplier = self.cell.input_multiplier
        self._projected_out_features = out_features * proj_multiplier
        fan_in = in_features + out_features if has_recurrent_connections else in_features
        self.linear = nnx.LinearGeneral(
            in_features,
            (out_features, proj_multiplier),
            rngs=rngs,
            kernel_init=get_initializer(kernel_initializer, fan_in, self._projected_out_features),
            use_bias=use_bias,
        )
        if has_recurrent_connections:
            self.recurrent = nnx.LinearGeneral(
                out_features,
                (out_features, proj_multiplier),
                rngs=rngs,
                use_bias=False,
                kernel_init=get_initializer(
                    rec_initializer, fan_in, self._projected_out_features
                ),
            )
        else:
            self.recurrent = None

    def __call__(self, inputs, init_state):
        I = self.linear(inputs)
        last_state, output = self.multi_step_forward(self, I, init_state)
        return output, last_state

    @staticmethod
    @nnx.scan(in_axes=(None, 1, nnx.Carry), out_axes=(nnx.Carry, 1))
    def multi_step_forward(layer, I, state):
        if layer.has_recurrent_connections:
            I_rec = layer.recurrent(state[..., -1])
            I = I + I_rec

        new_state, z_with_aux = layer.cell(
            I, state, layer.cell.params, layer.cell.hyperparams
        )
        z = z_with_aux[0]

        # carry, output
        return new_state, z

    def initialize_carry(self, input_shape):
        return self.cell.initialize_carry(input_shape)
