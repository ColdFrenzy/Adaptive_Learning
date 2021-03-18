from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import json
from utils.pre_compute_elo import compute_elo_difference
from config.custom_config import Config


def Connect4Eval(trainer, eval_workers):
    """custom evaluation function.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # TESTED, eval workers has different than trainer

    # We configured 2 eval workers in the training config.
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
