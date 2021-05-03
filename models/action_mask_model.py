from models import custom_models
from functools import reduce
from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from config.connect4_config import Connect4Config, Connect3Config

tf1, tf, tfv = try_import_tf()


class Connect4ActionMaskModel(TFModelV2):
    """Parametric action model that handles the dot product and masking.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(Connect4Config.WIDTH, Connect4Config.HEIGHT),
        action_embed_size=Connect4Config.N_ACTIONS,
        show_model=False,
        *args,
        **kwargs
    ):

        super(Connect4ActionMaskModel, self).__init__(
            obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,model_config=model_config, name=name# ,*args, **kwargs
        )

        # Obs_space has the wrong size. This is due to data dict preprocessor
        # that automatically flatten the original observation space.
        # retrieving the original observation space :
        hidden_layer_shapes = kwargs["hidden_layer_shapes"]
        self.use_conv = kwargs["use_conv"]
        print("preprocessed obs_space: ")
        print(obs_space)
        original_obs = obs_space.original_space.spaces["state"]
        print("The restored obs_space is: " + str(original_obs))
        
        if self.use_conv:
            in_shape = original_obs.shape
            self.base_model = custom_models.conv_dense_model(
                in_shape, num_outputs, "action_mask"
            )
            
        else:
            in_shape = original_obs.shape[0] * original_obs.shape[1]
            # inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
            # hidden_layer = tf.keras.layers.Dense(256, name="layer1", activation=tf.nn.relu)(
            #     inputs
            # )
            # self.out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
            #     hidden_layer
            # )
            # self.value_layer_out = tf.keras.layers.Dense(1, name="value", activation=None)(hidden_layer)
            # self.base_model = tf.keras.Model(inputs, [self.out_layer, self.value_layer_out], name=name)
    
            self.base_model = custom_models.dense_model(
                in_shape, hidden_layer_shapes, num_outputs, "action_mask"
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
        
        if self.use_conv:
            if len(obs_state.shape) < 3:
                obs_state = tf.expand_dims(obs_state, axis =-1)
                obs_state = tf.expand_dims(obs_state, axis = 0)
            elif len(obs_state.shape) == 3:
                obs_state = tf.expand_dims(obs_state, -1)
            else:
                pass
        else:
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

        action_logits, self._value_out = self.base_model(obs_state)

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
        return tf.reshape(self._value_out, [-1])





class Connect3ActionMaskModel(TFModelV2):
    """Parametric action model that handles the dot product and masking.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(Connect3Config.WIDTH, Connect3Config.HEIGHT),
        action_embed_size=Connect3Config.N_ACTIONS,
        show_model=False,
        *args,
        **kwargs
    ):

        super(Connect3ActionMaskModel, self).__init__(
            obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,model_config=model_config, name=name# ,*args, **kwargs
        )

        # Obs_space has the wrong size. This is due to data dict preprocessor
        # that automatically flatten the original observation space.
        # retrieving the original observation space :
        hidden_layer_shapes = kwargs["hidden_layer_shapes"]
        self.use_conv = kwargs["use_conv"]
        print("preprocessed obs_space: ")
        print(obs_space)
        original_obs = obs_space.original_space.spaces["state"]
        print("The restored obs_space is: " + str(original_obs))
        
        if self.use_conv:
            in_shape = original_obs.shape
            self.base_model = custom_models.conv_dense_model_connect3(
                in_shape, num_outputs, "action_mask"
            )
            
        else:
            in_shape = original_obs.shape[0] * original_obs.shape[1]
            # inputs = tf.keras.layers.Input(shape=(in_shape,), name="observations")
            # hidden_layer = tf.keras.layers.Dense(256, name="layer1", activation=tf.nn.relu)(
            #     inputs
            # )
            # self.out_layer = tf.keras.layers.Dense(num_outputs, name="out", activation=None)(
            #     hidden_layer
            # )
            # self.value_layer_out = tf.keras.layers.Dense(1, name="value", activation=None)(hidden_layer)
            # self.base_model = tf.keras.Model(inputs, [self.out_layer, self.value_layer_out], name=name)
    
            self.base_model = custom_models.dense_model(
                in_shape, hidden_layer_shapes, num_outputs, "action_mask"
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
        
        if self.use_conv:
            if len(obs_state.shape) < 3:
                obs_state = tf.expand_dims(obs_state, axis =-1)
                obs_state = tf.expand_dims(obs_state, axis = 0)
            elif len(obs_state.shape) == 3:
                obs_state = tf.expand_dims(obs_state, -1)
            else:
                pass
        else:
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

        action_logits, self._value_out = self.base_model(obs_state)

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
        return tf.reshape(self._value_out, [-1])