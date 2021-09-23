# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torchvision.models as models
from configs import Config
from cnnTR_model_v5 import cnnTR

def get_teachermodel(num_classes, configs):
    model = cnnTR(num_classes, configs)
    return model


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        # base_loss = self.base_criterion(outputs, labels)

        # if self.distillation_type == 'none':
        #     return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            per_frame_logits = self.teacher_model(inputs)
            teacher_outputs = torch.max(per_frame_logits, dim=2)[0]
            # print("teacher_outputs: ", teacher_outputs.shape)
        # print("outputs_kd: ", outputs_kd.shape)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
            # kd_label = torch.multinomial(teacher_outputs.softmax(-1), num_samples=1, replacement=False)
            # distillation_loss = F.cross_entropy(outputs_kd, kd_label.squeeze(-1))
        
        return distillation_loss
        # loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        # return loss


if __name__ == "__main__":
    a = torch.randn(10, 100)
    b = torch.randn(10, 100)
    T = 0.1
    distillation_loss = F.kl_div(
                F.log_softmax(a / T, dim=1),
                F.log_softmax(b / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / b.numel()
    print(distillation_loss)
    print(b.numel())

    distillation_loss = F.cross_entropy(a, b.argmax(dim=1))
    print(distillation_loss)
