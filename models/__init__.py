from ray.rllib.models import ModelCatalog
from models.action_mask_model import Connect4ActionMaskModel
from models.action_mask_q_model import Connect4ActionMaskQModel

ModelCatalog.register_custom_model("connect4_mask", Connect4ActionMaskModel)
ModelCatalog.register_custom_model("connect4_q_mask", Connect4ActionMaskQModel)
