from ray.rllib import Policy
import random


class RandomPolicy(Policy):
    """Policy that return a random action among the valid ones"""

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):

        action_space_len = self.action_space.n
        # first n elements of a single observation is the action mask
        random_acts = []
        for obs in obs_batch:
            action_mask = obs[:action_space_len]
            allowed_act = []
            for i, act in enumerate(action_mask):
                if act == 1:
                    allowed_act.append(i)

            random_acts.append(random.choice(allowed_act))
        return random_acts, [], {}

    def learn_on_batch(self, samples):
        """No learning."""
        # return {}
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

