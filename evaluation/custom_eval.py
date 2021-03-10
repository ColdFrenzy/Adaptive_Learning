import sys
import os
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
sys.path.insert(1, os.path.abspath(os.pardir))
from utils.pre_compute_elo import compute_elo_difference

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
    
    # Calling .sample() runs episodes on evaluation workers 
    # ray.get([w.sample.remote() for w in evaluation_workers])
    evaluation_workers.sample()
    
    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        local_worker=evaluation_workers, timeout_seconds=99999)
    
    metrics = summarize_episodes(episodes)
    
    player1_score = evaluation_workers.env.score["player1"]
    player2_score = evaluation_workers.env.score["player2"]           
    draws = evaluation_workers.env.num_draws
    metrics["player1_score"]  = player1_score
    metrics["minimax_score"]  = player2_score   
    metrics["number_of_draws"] = draws
    number_of_games = player1_score + player2_score + draws
    
    elo_diff = compute_elo_difference(player1_score,draws,number_of_games)
    metrics["elo_difference"]  = elo_diff
    


    return metrics