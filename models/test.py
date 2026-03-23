# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="in",
                 ):
        super().__init__()

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_scalar = adapter_scalar
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.non_linear_func = nn.ReLU()
        # 第五步，在Adapter中设置缩放矩阵
        self.down_rescale = nn.Parameter(torch.empty(1, self.down_size))
        self.down_bias = nn.Parameter(torch.empty(self.down_size))
        self.up_rescale = nn.Parameter(torch.empty(1, self.n_embd))
        self.up_bias = nn.Parameter(torch.empty(self.n_embd))

        nn.init.xavier_uniform_(self.down_rescale)
        nn.init.zeros_(self.down_bias)
        nn.init.xavier_uniform_(self.up_rescale)
        nn.init.zeros_(self.up_bias)




        self.dropout = dropout

    def forward(self, x, add_residual=True, residual=None, down_projection=None, up_projection=None):  # 768,64与64,768
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # down0 = self.down_proj(x)
        down = torch.matmul(x, down_projection * self.down_rescale) + self.down_bias


        down = self.non_linear_func(down)


        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        # up = self.up_proj(down)
        up = torch.matmul(down, up_projection * self.up_rescale) + self.up_bias

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output
