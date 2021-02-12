from ray.rllib.models import ModelCatalog
from models.action_mask_model import Connect4ActionMaskModel

ModelCatalog.register_custom_model("connect4_mask", Connect4ActionMaskModel)
