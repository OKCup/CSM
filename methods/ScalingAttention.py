import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import torch.nn.functional as F
from torch.nn import init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Scaling_Attention(nn.Module):
    def __init__(self, dim, planes=[144, 64, 64, 64, 144], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False, proj_kernel_size=(5,5)):
        super(Scaling_Attention, self).__init__()

        self.kernel_size = proj_kernel_size
        self.unfold = nn.Unfold(kernel_size=proj_kernel_size, padding=2)
        self.relu = nn.ReLU(inplace=True)

        self.attn = MultiDilatelocalAttention(dim, num_heads=6, qkv_bias=True, qk_scale=None,
                                              attn_drop=0., kernel_size=3, dilation=[1, 2, 3])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = nn.Identity()
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=0.5)

        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = self._create_conv_block(planes[0], planes[1], kernel_size=1, stride=(1, 1), padding=0,
                                                  is_3d=False)
        self.conv1 = self._create_conv_block(planes[1], planes[2], kernel_size=(1, self.ksize[2], self.ksize[3]),
                                             stride=stride, padding=padding1, is_3d=True, bias=bias)
        self.conv2 = self._create_conv_block(planes[2], planes[3], kernel_size=(1, self.ksize[2], self.ksize[3]),
                                             stride=stride, padding=padding1, is_3d=True, bias=bias)

        self.conv1x1_out = self._create_conv_block(planes[3], planes[4], kernel_size=1, stride=(1, 1), padding=0,
                                                   is_3d=False, final_layer=True)

        self.ELA = ELA_L(planes[-1], kernel_size=7)
        self.SE = SEAttention(planes[-1])


    def _create_conv_block(self, in_planes, out_planes, kernel_size, stride, padding, is_3d=False, bias=False,
                           final_layer=False):
        layers = []
        if is_3d:
            layers.append(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            layers.append(nn.BatchNorm3d(out_planes))
        else:
            layers.append(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            layers.append(nn.BatchNorm2d(out_planes))

        if not final_layer:
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x.permute(0, 3, 1, 2)
        b, c, h, w = x.shape
        #x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)  # b, cuv, h, w
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v

        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u * v)
        # x1 = self.ELA(x)
        # x2 = self.SE(x)
        # x = x+x1+x2
        x = self.conv1x1_in(x)

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)
        x = self.conv2(x)

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)
        return x


class ELA_L(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA_L, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.pad = self.kernel_size // 2
        self.conv = nn.Conv1d(self.channel, self.channel, kernel_size=self.kernel_size, padding=self.pad,
                              groups=self.channel // 8, bias=False)
        group = 16
        if (self.channel%16)!=0:
            group = 8
        self.gn = nn.GroupNorm(group, self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:input features with shape [b c h w]
        b, c, h, w = x.shape

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)# B, C, H, W
        #B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        #x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        # 使用新的张量存储结果
        out = []
        for i in range(self.num_dilation):
            out.append(self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2]).unsqueeze(0))  # B, H, W, C//num_dilation

        # 将结果拼接起来
        x = torch.cat(out, dim=0)
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x = x.reshape(B, C, H, W)
        return x
