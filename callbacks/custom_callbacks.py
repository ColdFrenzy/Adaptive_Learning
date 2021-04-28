from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.deprecation import deprecation_warning
from config.custom_config import Config
import numpy as np

class Connect4Callbacks(DefaultCallbacks):
    # set class attribute
    player1_score = 0.0
    player2_score = 0.0
    num_draws = 0.0
    score_diff = 0.0
    # we use the policy 
    weights_history = []
    def __init__(self, legacy_callbacks_dict=None):
        if legacy_callbacks_dict:
            deprecation_warning(
                "callbacks dict interface",
                "a class extending rllib.agents.callbacks.DefaultCallbacks",
            )
        self.legacy_callbacks = legacy_callbacks_dict or {}


    def reset_values(self):
        self.__class__.player1_score = 0.0
        self.__class__.player2_score = 0.0
        self.__class__.num_draws = 0.0
        self.__class__.score_diff = 0.0

    
    def load_values(self, previous_val):
        """
        This function is used to restore the values of a previous run, in order
        to continue the training 
        values: Dict
        """
        self.__class__.player1_score = previous_val["player1_score"]
        self.__class__.player2_score = previous_val["player2_score"]
        self.__class__.num_draws = previous_val["number_of_draws"]
        self.__class__.score_diff = previous_val["score_difference"]
        # self.opponent_policies = previous_val["opponent_policies"]

    def keep_best_weights(self,index_to_keep):
        """
            remove the oldest "num" weights
            :input: index_to_keep
                array with list of indices of weights to keep 
        """
        temp_weights = []
        for i in index_to_keep:
            temp_weights.append(self.__class__.weights_history[i])
        self.__class__.weights_history = temp_weights
          
            
    def add_weights(self,weight):
        """
            add weights to the current policy history 
        """
        self.__class__.weights_history.append(weight)
        
        
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
        # policies contains the same object that we get through worker.get_policy()
        # at the begin of every episode we dynamically set the weights of one
        # of the older policy 
        # TESTED this should actually work as a policy change 
        if len(self.__class__.weights_history) != 0:
            ind_vec = [ind for ind in range(len(self.__class__.weights_history))]
            index = np.random.choice(ind_vec)
            weight = self.__class__.weights_history[index]
            worker.get_policy("player2").set_weights(weight)
        # we can add data that we can use in other callbacks
        # episode.user_data["my_data"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
        everytime the env.step() function is called
         worker: "RolloutWorker",
         base_env: "BaseEnv",
         episode: "MultiAgentEpisode",
         env_index: int,
         **kwargs
        """
        # TEST DONE: the function worker.get_policy("player2").get_weights
        # returns the same weights of the self.__class__.weights_history[index]
        # where index is the one chosen in the on_episode_start callback
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
        # We add the final reward for players that have played the last match
        # 
        reward = {player: rw[-1] for player, rw in episode._agent_reward_history.items()}
        for player in reward:    
            episode.custom_metrics[player] = reward[player]
            
        # self.opponent_policies[episode._agent_to_policy["player2"]] += 1
        
        if reward["player1"] == 1.0:
            self.__class__.player1_score += 1.0 
        if reward["player2"] == 1.0:
            self.__class__.player2_score += 1.0 
        if reward["player1"] == 0.0:
            self.__class__.num_draws += 1.0 
        
        # an episode ends when done["__all__"] == True
        

    def on_sample_end(
            self,
            *,
            worker,
            samples,
            **kwargs):
        """
        worker: "RolloutWorker",
        samples: SampleBatch,
        **kwargs
        """
        # a player batches has a data dict with {action_dist,action_logp,actions,
        # advantages,agent_index,eps_id,obs,unroll_id,value_target,vf_preds}
        # p1_batches = samples.policy_batches["player1"].data
        # p2_batches = samples.policy_batches["player2"].data
        
        # Optional function that outputs game in a format readable to 
        # https://connect4.gamesolver.org/?pos=
        #TODO
                


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
        
        weights = policies("player1").get_weights
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

    def on_train_result(self, *, trainer, result, **kwargs):
        """
        self,
        *,
        trainer,
        result: dict,
        **kwargs
        
        """
        
        # The callback object inside on_train_result is different than 
        # the callback object inside the other functions. That's why i've 
        # used class attribute
        # create dict_keys if not exists
        # this got resetted_everytime
        result["custom_metrics"].setdefault("score_difference", 0.0)
        result["custom_metrics"].setdefault("number_of_draws", 0.0)
        result["custom_metrics"].setdefault("player1_score", 0.0)
        result["custom_metrics"].setdefault("player2_score", 0.0)
        result["custom_metrics"].setdefault("player1_win_rate",0.0)
        # remove useless metrics
        result["custom_metrics"].pop('player1_min', None)
        result["custom_metrics"].pop('player2_min', None)
        result["custom_metrics"].pop('player1_max', None)
        result["custom_metrics"].pop('player2_max', None)
        #result["custom_metrics"]["minimax_depth"] = trainer.get_policy("minimax").depth

        self.__class__.score_diff = self.__class__.player1_score - self.__class__.player2_score

        
        player1_score_this_iter = result["hist_stats"]["policy_player1_reward"].count(1.0)
        player1_draws_this_iter = result["hist_stats"]["policy_player1_reward"].count(0.0)
        episodes_this_iter = result["episodes_this_iter"]
        
        
        player1_win_rate = (player1_score_this_iter + player1_draws_this_iter/2)/episodes_this_iter
        
        result["custom_metrics"]["player1_win_rate"] = player1_win_rate
        #GLOBAL RESULTS ACROSS MULTIPLE ITERATIONS
        result["custom_metrics"]["score_difference"] = self.__class__.score_diff
        result["custom_metrics"]["number_of_draws"] = self.__class__.num_draws
        result["custom_metrics"]["player1_score"] = self.__class__.player1_score
        result["custom_metrics"]["player2_score"] = self.__class__.player2_score
 
        


class Connect4MiniMaxCallbacks(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict=None):
        if legacy_callbacks_dict:
            deprecation_warning(
                "callbacks dict interface",
                "a class extending rllib.agents.callbacks.DefaultCallbacks",
            )
        self.legacy_callbacks = legacy_callbacks_dict or {}



    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        pass

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):

        pass
    
    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        pass

    def on_sample_end(
            self,
            *,
            worker,
            samples,
            **kwargs):
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
        pass

    def on_train_result(self, *, trainer, result, **kwargs):
        """
        self,
        *,
        trainer,
        result: dict,
        **kwargs
        
        """
        result["custom_metrics"]["minimax_depth"] = trainer.get_policy("minimax").depth
        result["custom_metrics"]["player1_score"] = result["hist_stats"]["policy_player1_reward"].count(1.0)
        result["custom_metrics"]["player2_score"] = result["hist_stats"]["policy_minimax_reward"].count(1.0)
        result["custom_metrics"]["number_of_draws"] = result["hist_stats"]["policy_player1_reward"].count(0.0)
        result["custom_metrics"]["score_difference"] =  result["hist_stats"]["policy_player1_reward"].count(
            1.0
        ) - result["hist_stats"]["policy_minimax_reward"].count(1.0)


        

