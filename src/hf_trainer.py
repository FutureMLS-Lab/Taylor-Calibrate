from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Trainer

__all__ = ["KDTrainer"]


class KDTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self._teacher_on_gpu = False

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if not self._teacher_on_gpu:
            device = model.device if hasattr(model, 'device') else next(model.parameters()).device
            self.teacher_model = self.teacher_model.to(device)
            self._teacher_on_gpu = True
        loss = model.forward_kl(teacher=self.teacher_model, input_ids=inputs["input_ids"])
        return (loss, None) if return_outputs else loss
