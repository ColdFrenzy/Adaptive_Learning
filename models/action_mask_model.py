# from ray.rllib.models import ModelCatalog
import custom_models
from functools import reduce

from ray.rllib.models.tf import TFModelV2

# from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
# from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.utils.framework import try_import_tf
from config.custom_config import Config

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
        true_obs_shape=(Config.WIDTH, Config.HEIGHT),
        action_embed_size=Config.N_ACTIONS,
        show_model=False,
        *args,
        **kwargs
    ):

        super(Connect4ActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        # Obs_space has the wrong size. This is due to data dict preprocessor
        # that automatically flatten the original observation space.
        # retrieving the original observation space :
        print("preprocessed obs_space: ")
        print(obs_space)
        original_obs = obs_space.original_space.spaces["state"]
        print("The restored obs_space is: " + str(original_obs))
        in_shape = original_obs.shape[0] * original_obs.shape[1]

        # The observation space has already been flattered
        # self.inputs = tf.keras.layers.Input(shape=obs_space.shape[0]*obs_space.shape[1], name="observations")
        self.base_mode = custom_models.dense_model(in_shape, num_outputs, "action_mask")

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

        action_logits, _ = self.base_model(obs_state)

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
