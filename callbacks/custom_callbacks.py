# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:01:15 2021

@author: Francesco
"""

from ray.rllib.agents.callbacks import DefaultCallbacks


class Connect4Callbacks(DefaultCallbacks):
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
        try:
            print("Player 1 " + str(policies["player1"].__dict__))
        except:
            pass
        try:
            print("Player 2 " + str(policies["player2"].__dict__))
        except:
            pass

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
         worker: "RolloutWorker",
         base_env: "BaseEnv",
         episode: "MultiAgentEpisode",
         env_index: int,
         **kwargs
        """
        # print(info["episode"].__dict__)
        # print(info["env"].__dict__)

        # print the action distribution from our model. We use a try statement
        # at each step only one agent is present since the environment is turn
        # based
        try:
            print(
                "Player 1 action distribution: "
                + str(
                    episode.__dict__("_agent_to_last_pi_info")["player1"][
                        "action_dist_inputs"
                    ]
                )
            )
            print(
                "Player 1 action logp: "
                + str(
                    episode.__dict__("_agent_to_last_pi_info")["player1"]["action_logp"]
                )
            )
            print(
                "Player 1 action prob: "
                + str(
                    episode.__dict__["_agent_to_last_pi_info"]["player1"]["action_prob"]
                )
            )
        except:
            pass
        try:
            print(
                "Player 2 action distribution: "
                + str(
                    episode.__dict__["_agent_to_last_pi_info"]["player2"][
                        "action_dist_inputs"
                    ]
                )
            )
            print(
                "Player 2 action logp: "
                + str(
                    episode.__dict__["_agent_to_last_pi_info"]["player2"]["action_logp"]
                )
            )
            print(
                "Player 2 action prob: "
                + str(
                    episode.__dict__["_agent_to_last_pi_info"]["player2"]["action_prob"]
                )
            )
        except:
            pass

        return

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

        policy_obj = original_batches["pre_batch"][0]
        sample_obj = original_batches["pre_batch"][1]

        if agent_id == "player1":
            print("agent_id = {}".format(agent_id))
            print("episode = {}".format(episode.episode_id))
            # #print("on_postprocess_traj info = {}".format(info))
            # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
            print("actions = {}".format(sample_obj.columns(["actions"])))

        elif agent_id == "player2":
            print("agent_id = {}".format(agent_id))
            print("episode = {}".format(episode.episode_id))

            # #print("on_postprocess_traj info = {}".format(info))
            # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
            print("actions = {}".format(sample_obj.columns(["actions"])))
            # print('action_logs = {}'.format(sample_obj.columns(["action_dist_inputs"])))
            # print("on_postprocess_traj policy_obj = {}".format(policy_obj))
        return
