import torch
import torch.nn as nn
from ST_TR.TR_model_st_res import Model


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_hook()
       

    def register_hook(self):
        for layer in dict([*self.model.named_children()])['layers']: # 每层st_tr block
            layer.s_tr.register_forward_hook(
                lambda layer, _, output: print(f"{torch.norm(output, p=2)}")  # lambda 函数, 这里选取出空间transformer后的特征作为输出，查看其维度
            )
    

    def forward(self, x):
        return self.model(x)





