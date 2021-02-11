# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:53:15 2021

@author: Francesco
"""
# from ray.rllib.models import ModelCatalog
from functools import reduce

from ray.rllib.models.tf import TFModelV2
# from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
# from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.utils.framework import try_import_tf

import tensorflow as tf

class Connect4ActionMaskModel(TFModelV2):
    """Parametric action model that handles the dot product and masking."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(7, 6),
        action_embed_size=7,
        show_model=False,
        *args,
        **kwargs
    ):

        super(Connect4ActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        """ Obs_space has size (49,) but it should be (42,), this happens 
        because the gym space dict is automatically flattened from the preprocessor
        
        """
        # get real obs space, discarding action mask
        original_obs = obs_space.original_space.spaces["state"]
        print("The restored obs_space is: " + str(original_obs))

        # The observation space has already been flattered
        # self.inputs = tf.keras.layers.Input(shape=obs_space.shape[0]*obs_space.shape[1], name="observations")
        obs_dim= reduce(lambda x, y: x * y, original_obs.shape)
        inputs = tf.keras.layers.Input(shape=(obs_dim,), name="observations")
        hidden_layer = tf.keras.layers.Dense(256, name="layer1", activation=tf.nn.relu)(
            inputs
        )  # tf.nn.relu
        out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
            hidden_layer
        )
        self.base_model = tf.keras.Model(
            inputs, out_layer, name="action_mask"
        )

        if show_model == True:
            self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        """
        Args:
            input_dict (dict): dictionary of input tensors, including "obs",
                "obs_flat", "prev_action", "prev_reward", "is_training",
                "eps_id", "agent_id", "infos", and "t".
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths
        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, num_outputs], and the new RNN state.
        """

        obs_state = input_dict["obs"]["state"]
        action_mask = input_dict["obs"]["action_mask"]
        # print("The actual action mask is: ")
        # print(action_mask)
        # print("obs_state shape inside the model: " + str(obs_state.shape))
        # if a single example is passed
        if len(obs_state.shape) < 3:
            # print("obs_state tensor has rank: " + str(len(obs_state.shape)))
            obs_state = tf.reshape(
                obs_state, shape=(obs_state.shape[0] * obs_state.shape[1],)
            )
            # adding a dimension for the batch size if a single example is passed
            obs_state = tf.expand_dims(obs_state, 0)
        # if a batch is passed
        else:
            # print("obs_state tensor has rank: " + str(len(obs_state.shape)))
            obs_state = tf.reshape(
                obs_state,
                shape=(obs_state.shape[0], obs_state.shape[1] * obs_state.shape[2]),
            )

        action_logits = self.base_model(obs_state)

        # inf_mask return a 0 value if the action is valid and a big negative
        # value if it is invalid. Example:
        # [0, 0, -3.4+38, 0, -3.4+38, 0, 0]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        action_logits = tf.cast(action_logits, dtype=tf.float32)
        inf_mask = tf.cast(inf_mask, dtype=tf.float32)
        # The new logits have an extremely low value for invalid actions, that
        # is then cut to zero during the softmax computation
        new_action_logits = action_logits + inf_mask

        return new_action_logits, state

    def value_function(self):
        # return self.base_model.value_function()
        return self.value_layer_out()
