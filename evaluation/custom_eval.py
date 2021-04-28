from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import json
from utils.pre_compute_elo import compute_elo_difference, model_vs_minimax
from config.custom_config import Config


def Connect4Eval(trainer, eval_workers):
    """custom evaluation function. In this function we 
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # The functions Connect4Eval_2 gives some problem (wrong number of games
    # every eval iteration and bad minimax behaviour).
    # the idea here is to not use evaluation worker
    metrics = {}
    print("Model Evaluation")
    with open(Config.MINIMAX_DEPTH_PATH) as json_file:
        data = json.load(json_file)
        if data["minimax_depth"] == 5:
            return metrics
        
    number_of_games = Config.EVALUATION_NUMBER_OF_EPISODES
    depth = int(data["minimax_depth"])
    logger = trainer.get_policy("minimax").logger
    model = trainer.get_policy("player1").model
    elo_diff, p1_score, p2_score, draws = model_vs_minimax(
        model, depth, number_of_games, checkpoint=None, logger=logger
    )


    metrics["player1_score"] = p1_score
    metrics["minimax_score"] = p2_score
    metrics["number_of_draws"] = draws
    metrics["elo_difference"] = elo_diff
    metrics["minimax_depth"] = depth

    # if Config.ELO_DIFF_LWB <= elo_diff <= Config.ELO_DIFF_UPB:
    # if the model wins more than half of the games 
    if p1_score >= (number_of_games/2):
        trainer.save(Config.IMPORTANT_CKPT_PATH)
        data["minimax_depth"] += 1

        with open(Config.MINIMAX_DEPTH_PATH, "w") as json_file:
            json.dump(data, json_file)
            trainer.get_policy("minimax").depth = data["minimax_depth"]
    # if elo of our network is higher than elo upper bound, our network
    # is already stronger than this minimax stage, hence we skip this ckpt.
    # elif elo_diff > Config.ELO_DIFF_UPB:
    #     with open(Config.MINIMAX_DEPTH_PATH) as json_file:
    #         data = json.load(json_file)
    #     if data["minimax_depth"] == 5:
    #         return metrics
    #         data["skipped_depth"].append(data["minimax_depth"])
    #         data["minimax_depth"] += 1

    #     with open(Config.MINIMAX_DEPTH_PATH, "w") as json_file:
    #         json.dump(data, json_file)
    #         trainer.get_policy("minimax").depth = data["minimax_depth"]

    return metrics


def Connect4Eval_2(trainer, eval_workers):
    """custom evaluation function.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # TESTED, eval workers has different than trainer

    print("Model Evaluation")
    # evaluation_workers = eval_workers.remote_workers()
    # if no remote_workers use local worker for evaluation
    evaluation_workers = eval_workers.local_worker()

    evaluation_workers.env.reset_score()
    # Calling .sample() runs episodes on evaluation workers
    # ray.get([w.sample.remote() for w in evaluation_workers])
    evaluation_workers.sample()

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        local_worker=evaluation_workers, timeout_seconds=99999
    )

    metrics = summarize_episodes(episodes)

    player1_score = evaluation_workers.env.score["player1"]
    player2_score = evaluation_workers.env.score["player2"]
    draws = evaluation_workers.env.num_draws
    metrics["player1_score"] = player1_score
    metrics["minimax_score"] = player2_score
    metrics["number_of_draws"] = draws
    number_of_games = player1_score + player2_score + draws

    elo_diff = compute_elo_difference(player1_score, draws, number_of_games)
    metrics["elo_difference"] = elo_diff

    # when the elo_difference lies between a specific interval, we save the
    # checkpoint and we increase MiniMax depth for the next evaluation
    if Config.ELO_DIFF_LWB <= elo_diff <= Config.ELO_DIFF_UPB:
        trainer.save(Config.IMPORTANT_CKPT_PATH)
        with open(Config.MINIMAX_DEPTH_PATH,) as json_file:
            data = json.load(json_file)
            if data["minimax_depth"] == 5:
                return metrics
            data["minimax_depth"] += 1

        metrics["MINIMAX_DEPTH"] = data["minimax_depth"]
        with open(Config.MINIMAX_DEPTH_PATH, "w") as json_file:
            json.dump(data, json_file)
            evaluation_workers.get_policy("minimax").depth = data["minimax_depth"]
    # if elo of our network is higher than elo upper bound, our network
    # is already stronger than this minimax stage, hence we skip this ckpt.
    elif elo_diff > Config.ELO_DIFF_UPB:
        with open(Config.MINIMAX_DEPTH_PATH) as json_file:
            data = json.load(json_file)
        if data["minimax_depth"] == 5:
            return metrics
            data["skipped_depth"].append(data["minimax_depth"])
            data["minimax_depth"] += 1

        metrics["MINIMAX_DEPTH"] = data["minimax_depth"]
        with open(Config.MINIMAX_DEPTH_PATH, "w") as json_file:
            json.dump(data, json_file)
            evaluation_workers.get_policy("minimax").depth = data["minimax_depth"]

    return metrics
