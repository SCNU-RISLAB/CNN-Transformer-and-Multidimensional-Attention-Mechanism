from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()

        mip = max(4, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h


        return out

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class hswish(nn.Module):
    def forward(self,x):
        out=x*F.relu6(x+3,inplace=True)/6
        return out


class hsigmoid(nn.Module):
    def forward(self,x):
        out=F.relu6(x+3,inplace=True)/6
        return out

class SeModule(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


#线性瓶颈和反向残差结构
class mbBlock(nn.Module):
    def __init__(self, kernel_size, in_channel, expand_size, out_channel, nolinear, semodule, stride):
        super(mbBlock, self).__init__()
        self.stride = stride
        self.se = semodule
        #1*1展开卷积
        self.conv1 = nn.Conv2d(in_channel, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        #3*3（或5*5）深度可分离卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        #1*1投影卷积
        self.conv3 = nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #注意力模块
        if self.se != None:
            out = self.se(out)
        #残差链接
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.block=mbBlock(7,128,512,128,hswish(), SeModule(128), 1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x=self.block(x)

        x = x.flatten(2).permute(0, 2, 1)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

        self.ca=CA(dim,self.qk_dim,16)

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape

        xq=x.permute(0, 2, 1).reshape(B, C, H, W)

        xq=self.ca(xq)

        xq = xq.reshape(B, self.qk_dim, -1).permute(0, 2, 1)

        q =xq.reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1,3)  # b,head,n,qkdim/head

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # b,n',c
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1,
                                                                                                 3)  # b,head,n',qkdim/head
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # b,head,n',c/head
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # b,head,n,qkdim/head
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # b,head,n,c/head

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos  # b,head,n,n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # b,head,n,c/head->b,n,c
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.DWConv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, groups=dim, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.PWConv = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )


    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape  # x是b,h*w,c
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        #LLC
        x = x +self.DWConv(x)
        x = self.PWConv(x)

        x = x.flatten(2).permute(0, 2, 1)  # 变回b,h*w,c

        #CA-LMAM
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))

        #SE-LMAM
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(80, 251), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)#将输入转换为元组，比如224->(224,224)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # 把h w合并并且交换c和h*w->b,h*w,c
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class LCT(nn.Module):
    def __init__(self, img_size=(80, 251), num_classes=1000, embed_dims=46,fc_dim=1280,
                 num_heads=1, mlp_ratios=3.6, qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=2, qk_ratio=1, sr_ratios=8, dp=0.1, noclass=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_patches = img_size[0] * img_size[1]

        self.relative_pos = nn.Parameter(torch.randn(
            num_heads, self.num_patches,
            self.num_patches // sr_ratios // sr_ratios))


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        cur = 0
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios)
            for i in range(depths)])

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.noclass = noclass
        # Classifier head
        self._fc = nn.Conv2d(embed_dims, fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.size()
        x = x.reshape(B, C, W * H)
        x = x.permute(0, 2, 1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W, self.relative_pos)
        x = x.reshape(B, C, H, W)

        if self.noclass:
            return x
        else:
            x = self._fc(x)
            x = self._bn(x)
            x = self._swish(x)
            x = self._avg_pooling(x).flatten(start_dim=1)
            x = self._drop(x)
            x = self.pre_logits(x)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.noclass:
            return x
        else:
            x = self.head(x)
            return x

