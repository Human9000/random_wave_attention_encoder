import torch
from torch import nn
from einops import rearrange
import numpy as np
from ctypes import CDLL, c_bool, c_float, c_void_p
import os


# 加载dll
libwre = CDLL("dll/wre_bnlved.dll")

#
def make_wre(b, l, gama):  # b, l
    n = int(gama*l)
    if n<2: n=2
    
    v0 = np.ones((b, n-1), dtype=c_bool)
    v1 = np.zeros((b, l-n-2), dtype=c_bool)
    v2 = np.ones((b, 1), dtype=c_bool)
    v = np.concatenate((v0,v1,v2),axis=1) 
    
    # 数据类型处理
    n = int(v[0].sum())  # c 中采用的 int 类型
    e = np.zeros((b, l, n), dtype=c_float)  # c 中采用的 float类型
    d = np.zeros((b, n, l), dtype=c_float)  # c 中采用的 float类型
    # 调用C语言，循环处理生成
    print(b,n,l)
    libwre.random_en_de(b, n, l,  # 指定后面多维数组的大小
                        v.ctypes.data_as(c_void_p),  # mask 的指针
                        e.ctypes.data_as(c_void_p),  # encoder
                        d.ctypes.data_as(c_void_p))  # decoder
    return e, d

# 残差链接的双卷积层
class ResDoubleConv3d(torch.nn.Module):
    def __init__(self, cin, cout, kernal, stride, padding) -> None:
        super().__init__()
        self.c1 = torch.nn.Conv3d(cin, cout, kernal, stride, padding)
        self.c2 = torch.nn.Conv3d(cout, cout, kernal, stride, padding)
        self.relu = torch.nn.ReLU()
        self.normal = torch.nn.BatchNorm3d(cout)

    def forward(self, x):
        y1 = self.c1(x)
        y2 = self.c2(self.relu(self.normal(y1)))
        return y1+y2


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):  # b n c
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SvdWaveAttention(torch.nn.Module):
    def __init__(self, c, q=2, heads=8, dim_head=64) -> None:
        super().__init__()
        self.q = q
        self.c = c
        self.cu3 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.cv3 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.cu2 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.cv2 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.cu1 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.cv1 = ResDoubleConv3d(c, c, (3, 3, 1), 1, (1, 1, 0))  # 位置解码
        self.P = []
        self.attn = Attention(q, heads, dim_head)

    def encode(self, x):
        u3, s3, v3 = torch.svd_lowrank(x, q=self.q)  # 降序svd
        u2, s2, v2 = torch.svd_lowrank(x.transpose(-3, -2), q=self.q)  # 降序svd
        u1, s1, v1 = torch.svd_lowrank(x.transpose(-3, -1), q=self.q)  # 降序svd

        # 参数 P
        self.P = [
            self.cu3(u3),
            self.cv3(v3),
            self.cu2(u2),
            self.cv2(v2),
            self.cu1(u1),
            self.cv1(v1),
        ]

        # 波特征 S
        s3 = rearrange(s3, 'b c n w -> (b c) n w')
        s2 = rearrange(s2, 'b c n w -> (b c) n w')
        s1 = rearrange(s1, 'b c n w -> (b c) n w')

        return [s3, s2, s1]

    def attention(self, s3s2s1):
        s3, s2, s1 = s3s2s1
        n3, n2, n1 = s3.shape[-2], s2.shape[-2], s1.shape[-2]
        # 多维合并自注意力
        s = self.attn(torch.cat(s3s2s1, dim=-2))
        # 多维波特征分解
        s3 = s[..., :n3, :]
        s2 = s[..., n3:-n1, :]
        s1 = s[..., -n1:, :]

        return [s3, s2, s1]

    def decode(self, s3s2s1):
        s3, s2, s1 = [rearrange(s,'(b c) l q -> b c l q',c=self.c) for s in s3s2s1]

        # SVD解码
        u3, v3, u2, v2, u1, v1 = self.P  # 位置解码参数
        print(u3.shape, s3.shape, v3.shape)
        y3 = u3@torch.diag_embed(s3)@v3.mT
        y2 = (u2@torch.diag_embed(s2)@v2.mT).transpose(-3, -2)
        y1 = (u1@torch.diag_embed(s1)@v1.mT).transpose(-3, -1)
        y = (y3 + y2 + y1) / 3
        return y


class WaveRandomEncoder(nn.Module):
    def __init__(self, gama) -> None:
        super().__init__()
        self.EN = False
        self.DE = False
        self.gama = gama**(1/3)
        print(gama, self.gama)

    def make_en_de(self, s3s2s1):
        s3, s2, s1 = s3s2s1 

        EN3, DE3 = make_wre(*s3.shape[:2], gama)
        EN2, DE2 = make_wre(*s3.shape[:2], gama)
        EN1, DE1 = make_wre(*s3.shape[:2], gama)

        self.EN321 = [torch.FloatTensor(EN3, device=s3.device),
                      torch.FloatTensor(EN2, device=s2.device),
                      torch.FloatTensor(EN1, device=s1.device)
                      ]
        self.DE321 = [torch.FloatTensor(DE3, device=s3.device),
                      torch.FloatTensor(DE2, device=s2.device),
                      torch.FloatTensor(DE1, device=s1.device)
                      ]

    def encode(self, s321): 
        return [(s.mT@e).mT for s,e in zip(s321, self.EN321)]

    def decode(self, s321): 
        return [(s.mT@e).mT for s,e in zip(s321, self.DE321)]


if __name__ == '__main__':
    b, c = 2, 2 # batchsize , channle
    d, w, h = 100, 100, 100
    gama = 0.2 # wre模块的特征保留系数
    q = 6 # 波特征数
    x = torch.randn((b, c, d, w, h))  # batchsize*chanel, length, word
    swa = SvdWaveAttention(c, q)
    wre = WaveRandomEncoder(gama)

    s = swa.encode(x)  # 转化成波
    print('swa波:', s[0].shape, s[1].shape, s[2].shape)
    wre.make_en_de(s)  # 生成随机波编解码器
    print('wre编解码器', wre.EN321[0].shape)
    rs = wre.encode(s)  # 随机波
    print('rs波:',rs[0].shape)
    rsa = swa.attention(rs)  # 随机波注意力
    print('res波注意力:', rsa[0].shape)
    sa = wre.decode(rsa)  # 重建的波注意力
    print('sa波注意力:',sa[0].shape)
    va = swa.decode(sa)  # 重建的体素注意力
    print('va注意力:',va[0].shape)
