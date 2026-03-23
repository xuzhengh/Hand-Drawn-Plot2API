# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


import math
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

# k, t, c, SE, HS, s
cfgs = [3, 2, 64, 1, 0, 1]



class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()

        self.identity = stride == 1 and inp == oup
        assert (self.identity), "identity connection required when stride is 1 and inp == oup"
        
        self.conv1=RepVGGDW(inp)
        self.se=SqueezeExcite(inp, 0.25) if use_se else nn.Identity()
        self.token_mixer = nn.Sequential(self.conv1,self.se)
        
        self.conv2=Conv2d_BN(inp, hidden_dim, 1, 1, 0)
        self.gelu=nn.GELU() if use_hs else nn.GELU()
        self.conv3=Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0)
        self.channel_mixer = Residual(nn.Sequential(self.conv2,self.gelu,self.conv3))

    def forward(self, x):

        out = self.channel_mixer(self.token_mixer(x))

        return out



class Adapter(nn.Module):
    def __init__(self,
                 config=None,  # 配置
                 d_model=None,  # 嵌入维度
                 bottleneck=None,  # 瓶颈层的大小
                 dropout=0.0,  # Dropout 比例
                 init_option="bert",  # 权重初始化方法选项（默认为BERT方式）
                 adapter_scalar="learnable_scalar",  # 缩放因子的类型
                 adapter_layernorm_option="in"):  # 是否层归一化
        super().__init__()

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_scalar = adapter_scalar

        self.adapter_layernorm_option = adapter_layernorm_option

        # self.adapter_layer_norm_before = None
        # if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
        #     self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        
        self.nonlinear = F.gelu
        
        self.dropout = dropout
        
        k, t, c, use_se, use_hs, s=cfgs
        output_channel = _make_divisible(c, 8)
        exp_size = _make_divisible( c * t, 8)
        self.block=RepViTBlock(c, exp_size, output_channel, k, s, use_se, use_hs)
        
        self.norm = nn.LayerNorm(self.n_embd)
        self.gamma = nn.Parameter(torch.ones(self.n_embd) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(self.n_embd))
        
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        # if self.adapter_layernorm_option == 'in':
        #     x = self.adapter_layer_norm_before(x)
        x = self.norm(x) * self.gamma + x * self.gammax
        

        down = self.down_proj(x)
        
        cls_token = down[:, 0:1, :]
        down = down[:, 1:, :]
        
        b, n, c = down.shape
        h = w = int(math.sqrt(n))
        
        down = down.reshape(b, h, w, c).permute(0, 3, 1, 2)
        down = self.block(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        down = down.permute(0, 2, 3, 1).reshape(b, n, c)
        
        down = torch.cat((cls_token, down), dim=1)
        
        down = self.nonlinear(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        up = self.up_proj(down)

        up = up * self.scale

        # if self.adapter_layernorm_option == 'out':
        #     up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

