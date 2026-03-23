# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    # 初始化方法：用于设置模型的各种参数和网络层
    def __init__(self,
                 config=None,  # 模型的配置
                 d_model=None,  # 嵌入维度
                 bottleneck=None,  # 瓶颈层的大小
                 dropout=0.0,  # Dropout 比例
                 init_option="bert",  # 权重初始化方法选项（默认为BERT方式）
                 adapter_scalar="learnable_scalar",  # 缩放因子的类型（默认为可学习的缩放因子）
                 adapter_layernorm_option="in"):  # 是否在输入或输出应用LayerNorm（层归一化）
        super().__init__()

        # 根据传入的配置或默认值设置嵌入维度（d_model）和瓶颈层大小（down_size）
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # 设置适配器的缩放因子类型
        self.adapter_scalar = adapter_scalar

        # 设置是否使用层归一化
        self.adapter_layernorm_option = adapter_layernorm_option

        # 如果需要，在适配器前应用层归一化
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # 如果缩放因子是可学习的，则创建一个可学习的参数
        if adapter_scalar == "learnable_scalar":
            print("okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")  # 输出确认信息
            self.scale = nn.Parameter(torch.ones(1))  # 初始化为1的可学习参数
        else:
            self.scale = float(adapter_scalar)  # 否则使用固定的缩放因子值

        # 定义降维层（从n_embd到down_size）
        self.down_proj = nn.Linear(self.n_embd, self.down_size)

        # 定义非线性激活函数，使用ReLU
        self.non_linear_func = nn.ReLU()

        # 定义升维层（从down_size回到n_embd）
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # 设置Dropout的比例
        self.dropout = dropout

        # 权重初始化方法选择，如果选中“lora”则执行相关初始化
        if init_option == "bert":
            raise NotImplementedError  # BERT初始化方式尚未实现
        elif init_option == "lora":
            # 使用Kaiming Uniform初始化（适合ReLU激活的层）
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)  # 上升层权重初始化为零
                nn.init.zeros_(self.down_proj.bias)  # 降维层偏置初始化为零
                nn.init.zeros_(self.up_proj.bias)  # 上升层偏置初始化为零
        #print(f"111: {adapter_scalar}")
        #print(f"222: {adapter_layernorm_option}")
    
    # 前向传播函数
    def forward(self, x, add_residual=True, residual=None):
        # 如果没有传入残差，则设置为输入x
        residual = x if residual is None else residual

        # 如果选中“in”选项，则在输入层进行LayerNorm
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # 通过降维层处理输入
        # print(f"降维前形状: {x.shape}")
        down = self.down_proj(x)

        # 经过ReLU激活
        # print(f"ReLU 输入形状: {down.shape}")
        down = self.non_linear_func(down)
        # print(f"ReLU 输出形状: {down.shape}")

        # 经过Dropout操作（防止过拟合）
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        # 通过升维层将特征还原到原始维度
        up = self.up_proj(down)

        # 缩放输出（乘以缩放因子）
        up = up * self.scale

        # 如果选中“out”选项，则在输出层进行LayerNorm
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        # 如果需要残差连接，则将输入的残差加到最终输出
        if add_residual:
            output = up + residual
        else:
            output = up

        # 返回输出
        return output
