from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.deprecation import deprecation_warning


class Connect4Callbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict=None):
        if legacy_callbacks_dict:
            deprecation_warning(
                "callbacks dict interface",
                "a class extending rllib.agents.callbacks.DefaultCallbacks",
            )
        self.legacy_callbacks = legacy_callbacks_dict or {}
        # some values for tensorboard
        self.player1_score = 0.0
        self.player2_score = 0.0
        self.num_draws = 0.0
        self.score_diff = 0.0

    def reset_values(self):
        self.player1_score = 0.0
        self.player2_score = 0.0
        self.num_draws = 0.0
        self.score_diff = 0.0

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        """
        Useful to check if everything was setted correctly 
        worker: RolloutWorker, 
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode, 
        env_index: int,
        **kwargs
        
        """
        # we can add data that we can use in other callbacks
        # episode.user_data["my_data"] = []
        pass

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
         worker: "RolloutWorker",
         base_env: "BaseEnv",
         episode: "MultiAgentEpisode",
         env_index: int,
         **kwargs
        """
        # everytime the env.step() function is called

        pass

    def on_postprocess_traj(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        """
        worker: RolloutWorker,
        episode: MultiAgentEpisode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs

        """

        # policy_obj = original_batches["pre_batch"][0]
        # sample_obj = original_batches["pre_batch"][1]

        # if agent_id == "player1":
        #     print("agent_id = {}".format(agent_id))
        #     print("episode = {}".format(episode.episode_id))
        #     # #print("on_postprocess_traj info = {}".format(info))
        #     # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        #     print("actions = {}".format(sample_obj.columns(["actions"])))

        # elif agent_id == "player2":
        #     print("agent_id = {}".format(agent_id))
        #     print("episode = {}".format(episode.episode_id))

        #     # #print("on_postprocess_traj info = {}".format(info))
        #     # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        #     print("actions = {}".format(sample_obj.columns(["actions"])))
        #     # print('action_logs = {}'.format(sample_obj.columns(["action_dist_inputs"])))
        #     # print("on_postprocess_traj policy_obj = {}".format(policy_obj))
        # return
        pass

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        """
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
        
        """
        # an episode ends when done["__all__"] == True

    def on_train_result(self, *, trainer, result, **kwargs):
        """
        self,
        *,
        trainer,
        result: dict,
        **kwargs
        
        """

        # create dict_keys if not exists
        # this got resetted_everytime
        result["custom_metrics"].setdefault("score_difference", 0.0)
        result["custom_metrics"].setdefault("number_of_draws", 0.0)
        result["custom_metrics"].setdefault("player1_score", 0.0)
        result["custom_metrics"].setdefault("player2_score", 0.0)

        self.player1_score += result["hist_stats"]["policy_player1_reward"].count(1.0)
        self.player2_score += result["hist_stats"]["policy_player2_reward"].count(1.0)
        self.num_draws += result["hist_stats"]["policy_player1_reward"].count(0.0)
        self.score_diff += result["hist_stats"]["policy_player1_reward"].count(
            1.0
        ) - result["hist_stats"]["policy_player2_reward"].count(1.0)

        result["custom_metrics"]["score_difference"] = self.score_diff
        result["custom_metrics"]["number_of_draws"] = self.num_draws
        result["custom_metrics"]["player1_score"] = self.player1_score
        result["custom_metrics"]["player2_score"] = self.player2_score
