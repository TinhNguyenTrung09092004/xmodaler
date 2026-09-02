# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_losses, build_rl_losses, add_loss_config

from .label_smoothing import LabelSmoothing
from .reward_criterion import RewardCriterion
from .semcomphder_loss import SemComphderLoss