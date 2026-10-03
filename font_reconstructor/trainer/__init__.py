from .base_trainer import BaseTrainer, strip_data_parallel_prefix
from .trainer import Trainer

__all__ = ['BaseTrainer', 'Trainer', 'strip_data_parallel_prefix']
