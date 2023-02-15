import torch
from torch import nn
from einops import rearrange
import numpy as np
from ctypes import CDLL, c_bool, c_float, c_void_p
import os

if not os.path.isfile("model/wre_bnlved.dll"):
    os.system('gcc -fPIC -shared -o model/wre_bnlved.dll model/wre_bnlved.cpp')
    
# 加载dll
libwre = CDLL("model/wre_bnlved.dll")

# wre 生成器
def make_wre(b, l, gama):  # b, l
    n = int(gama*l)
    if n<2: n=2 
    # 定义变量+ 数据类型处理
    v = np.zeros((b, l), dtype=c_bool) # c 中采用的 bool类型
    e = np.zeros((b, l, n), dtype=c_float)  # c 中采用的 float类型
    d = np.zeros((b, n, l), dtype=c_float)  # c 中采用的 float类型
    # 调用c函数
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

# 自注意力机制
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

# 奇异值分解的波化注意力
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
        
        y3 = u3@torch.diag_embed(s3)@v3.transpose(-1,-2)
        y2 = (u2@torch.diag_embed(s2)@v2.transpose(-1,-2)).transpose(-3, -2)
        y1 = (u1@torch.diag_embed(s1)@v1.transpose(-1,-2)).transpose(-3, -1)
        y = (y3 + y2 + y1) / 3
        return y

# 波随机编码器
class WaveRandomEncoder(nn.Module):
    def __init__(self, gama) -> None:
        super().__init__()
        self.EN = False
        self.DE = False
        self.gama3 = gama**(1/3) 

    def make_en_de(self, s3s2s1):
        s3, s2, s1 = s3s2s1 

        EN3, DE3 = make_wre(*s3.shape[:2], self.gama3)
        EN2, DE2 = make_wre(*s3.shape[:2], self.gama3)
        EN1, DE1 = make_wre(*s3.shape[:2], self.gama3)

        self.EN321 = [torch.FloatTensor(EN3 ).to(s3.device),
                      torch.FloatTensor(EN2 ).to(s3.device),
                      torch.FloatTensor(EN1 ).to(s3.device)
                      ]
        self.DE321 = [torch.FloatTensor(DE3 ).to(s3.device),
                      torch.FloatTensor(DE2 ).to(s3.device),
                      torch.FloatTensor(DE1 ).to(s3.device)
                      ]

    def encode(self, s321): 
        return [(s.transpose(-1,-2)@e).transpose(-1,-2) for s,e in zip(s321, self.EN321)]

    def decode(self, s321): 
        return [(s.transpose(-1,-2)@e).transpose(-1,-2) for s,e in zip(s321, self.DE321)]


    
class RWAE(nn.Module):
    def __init__(self, c, q, gama) -> None:
        super().__init__()
        self.swa = SvdWaveAttention(c, q)
        self.wre = WaveRandomEncoder(gama)

    def forward(self, x, LOG=False):
        s = self.swa.encode(x)  # 转化成波
        if LOG:print('S波:', s[0].shape, s[1].shape, s[2].shape)
        self.wre.make_en_de(s)  # 生成随机波编解码器
        if LOG:print('EN随机编码器', self.wre.EN321[0].shape)
        rs = self.wre.encode(s)  # 随机波
        if LOG:print('RS随机波:',rs[0].shape)
        rsa = self.swa.attention(rs)  # 随机波注意力
        if LOG:print('RSA随机波注意力:', rsa[0].shape)
        sa = self.wre.decode(rsa)  # 重建的波注意力
        if LOG:print('SA波注意力:',sa[0].shape)
        va = self.swa.decode(sa)  # 重建的体素注意力
        if LOG:print('VA体素注意力:',va[0].shape)
        return va + x


if __name__ == '__main__':
    b, c = 2, 2 # batchsize , channle
    d, w, h = 20, 20, 20
    gama = 0.1 # wre模块的特征保留系数
    q = 6 # 波特征数
    x = torch.randn((b, c, d, w, h)).cuda(0)  # batchsize*chanel, length, word
    rwae = RWAE(c,q,gama).cuda(0)
    rwae(x,LOG=True)
     
