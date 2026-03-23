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
        """
        RepViT
        inp: 输入通道数
        hidden_dim: 隐藏层通道数
        oup: 输出通道数
        kernel_size: 卷积核大小
        stride: 步幅
        use_se: 是否使用 Squeeze-Excite 模块
        use_hs: 是否使用 GELU 激活函数
        """
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2], "stride must be 1 or 2"

        # 输入通道=输出通道且步幅=1时
        self.identity = stride == 1 and inp == oup

        # 确保隐藏层通道数是输入通道数的两倍
        assert (hidden_dim == 2 * inp), "hidden_dim must be 2 * inp"
        if stride == 2:
            # 如果步幅为2，构建token mixer
            self.token_mixer = nn.Sequential(
                # 深度卷积 + 批归一化
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),

                # 如果使用 Squeeze-Excite，则添加 SqueezeExcite 模块，否则使用恒等映射
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),

                # 将通道数从 inp 映射到 oup
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )

            # channel mixer
            self.channel_mixer = Residual(nn.Sequential(
                # 逐点卷积，将通道数从 oup 映射到 2 * oup
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),

                # 如果使用 GELU 激活函数，则添加 GELU，否则也添加 GELU（此处可能有误，通常可能是另一种激活函数）
                nn.GELU() if use_hs else nn.GELU(),

                # 逐点线性卷积，将通道数从 2 * oup 映射回 oup，并初始化批归一化层的权重为0
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            # 如果步幅为1，且使用身份连接
            assert (self.identity), "identity connection required when stride is 1 and inp == oup"

            # token mixer
            self.token_mixer = nn.Sequential(
                # RepVGG深度卷积模块
                RepVGGDW(inp),

                # 是否使用SqueezeExcite，如果use_se参数为1就使用，不为1就使用恒等映射
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )

            # channel mixer
            self.channel_mixer = Residual(nn.Sequential(
                # 将通道数从inp映射到hidden_dim
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),

                # 不管use_hs是什么都GELU？
                nn.GELU() if use_hs else nn.GELU(),

                # 将通道数从hidden_dim映射回oup，并初始化批归一化层的权重为0
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))
        # print("输入通道数：  ",inp)
        # print("隐藏层通道数：", hidden_dim)
        # print("输出通道数：  ", oup)
        # print("卷积核大小：  ", kernel_size)
        # print("步幅：        ", stride)
        # print("是否使用 Squeeze-Excite 模块：", use_se)
        # print("是否使用 GELU 激活函数：", use_hs)
        # print("\n")

    def forward(self, x):
        # print("repvit输入前：",x.shape)
        # 先通过 Token Mixer，然后通过 Channel Mixer
        out = self.channel_mixer(self.token_mixer(x))
        # print("repvit输入后：", out.shape)
        return out
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        # self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        # self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
        self.se = SqueezeExcite(in_features, 0.25)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        # conv2_x = self.conv2(x)
        # conv3_x = self.conv3(x)

        x = conv1_x + identity

        #qrqrqr
        x = self.se(x)
        
        identity = x

        x = self.projector(x)

        return identity + x

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

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)

        # ReLU
        # self.non_linear_func = nn.ReLU()
        self.nonlinear = F.gelu
        # self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(self.n_embd)
        self.gamma = nn.Parameter(torch.ones(self.n_embd) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(self.n_embd))

        self.adapter_conv = MonaOp(64)
        
        
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

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

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        
        cls_token = down[:, 0:1, :]
        down = down[:, 1:, :]
        
        b, n, c = down.shape
        h = w = int(math.sqrt(n))
        
        down = down.reshape(b, h, w, c).permute(0, 3, 1, 2)
        down = self.adapter_conv(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        down = down.permute(0, 2, 3, 1).reshape(b, n, c)
        
        down = torch.cat((cls_token, down), dim=1)
        
        down = self.nonlinear(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

