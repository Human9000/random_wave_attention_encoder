import torch
from torch import nn
from einops import rearrange
import numpy as np
from ctypes import CDLL, c_bool, c_float, c_void_p
import os
from torch.nn import functional as F

from torchvision.transforms import functional as TF

if not os.path.isfile("model/wre_bnlved.dll"):
    os.system('gcc -fPIC -shared -o model/wre_bnlved.dll model/wre_bnlved.cpp')
    
# 加载dll
libwre = CDLL("model/wre_bnlved.dll")

# wre 生成器
def make_wre(b, l, gama):  # b, l
    n = int(gama*l)
    if n<2: n=2
    # 定义变量+ 数据类型处理
    v = np.zeros((b, l), dtype=c_bool)
    e = np.zeros((b, l, n), dtype=c_float)  # c 中采用的 float类型
    d = np.zeros((b, n, l), dtype=c_float)  # c 中采用的 float类型
    # 调用c函数
    libwre.random_en_de(b, n, l,  # 指定后面多维数组的大小
                        v.ctypes.data_as(c_void_p),  # mask 的指针
                        e.ctypes.data_as(c_void_p),  # encoder
                        d.ctypes.data_as(c_void_p))  # decoder
    return e, d

# wre 生成器
def make_mask_wre(mask, fact=4, continuity=False):  # b, l
    # 定义变量+ 数据类型处理
    l = mask.shape[0]
    if continuity:
        s,e = 0, 0
        p = -1
        for i in range(l):
            if mask[i] == True:
                if p == -1:
                    p = i
            elif (p !=-1):
                if (i - p > e - s):
                    s,e = p,i
                p = -1
        if s>5:
            s-=5
        if e<l-5: e+=5
        mask[:s] = False
        mask[e:] = False
        mask[s:e] = True
    mask[::fact] = True
    n = int(mask.sum())
    v = mask.detach().cpu().numpy().astype(c_bool)
    e = np.zeros((l, n), dtype=c_float)  # c 中采用的 float类型
    d = np.zeros((n, l), dtype=c_float)  # c 中采用的 float类型
    # 调用c函数
    libwre.mask_en_de(n, l,  # 指定后面多维数组的大小
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
    def __init__(self, dim, heads=8, dim_head=8):
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
        self.attn = Attention(c*q*q, heads, dim_head)

    def encode(self, x): 
        b,c = x.shape[:2]
#         x3 = F.adaptive_avg_pool2d(x, (self.q, self.q))
#         x2 = F.adaptive_avg_pool2d(x.transpose(-3, -2),(self.q,self.q))
#         x1 = F.adaptive_avg_pool2d(x.transpose(-3, -1),(self.q,self.q))
#         s3 = x3.reshape((b, c, -1)).transpose(-2, -1)
#         s2 = x2.reshape((b, c, -1)).transpose(-2, -1)
#         s1 = x1.reshape((b, c, -1)).transpose(-2, -1)
        
        x3 = x.mean(dim=[0,1,2], keepdim=True)
        x2 = x.mean(dim=[0,1,3], keepdim=True).transpose(2, 3)
        x1 = x.mean(dim=[0,1,4], keepdim=True).transpose(2, 4)
        
        u3, s3, v3 = torch.svd_lowrank(x3, q=self.q)  # 降序svd
        u2, s2, v2 = torch.svd_lowrank(x2, q=self.q)  # 降序svd
        u1, s1, v1 = torch.svd_lowrank(x1, q=self.q)  # 降序svd
        
        s3 = u3.transpose(-2, -1)@x@v3 
        s2 = u2.transpose(-2, -1)@x@v1 
        s1 = u1.transpose(-2, -1)@x@v1 
#         # 参数 P
#         self.P = [
#             self.cu3(u3),
#             self.cv3(v3),
#             self.cu2(u2),
#             self.cv2(v2),
#             self.cu1(u1),
#             self.cv1(v1),
#         ]
        # 参数 P
        self.P = [
            u3,
            v3.transpose(-2, -1),
            u2,
            v2.transpose(-2, -1),
            u1,
            v1.transpose(-2, -1),
        ]
        
#         # 波特征 S
        s3 = rearrange(s3, 'b c n q1 q2 -> b n (c q1 q2)')
        s2 = rearrange(s2, 'b c n q1 q2 -> b n (c q1 q2)')
        s1 = rearrange(s1, 'b c n q1 q2 -> b n (c q1 q2)') 
        
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
#         s3, s2, s1 = [rearrange(s,'b l q2 -> b q2 l',c=self.c).unsqueeze(-1).unsqueeze(-1) for s in s3s2s1]
        s3, s2, s1 = [rearrange(s,'b l (c q1 q2) -> b c l q1 q2', c=self.c, q1=self.q, q2=self.q) for s in s3s2s1]

        # SVD解码
        u3, v3, u2, v2, u1, v1 = self.P  # 位置解码参数
        
        y3 = u3@ s3@v3 
        y2 = (u2@ s2@v2).transpose(-3, -2)
        y1 = (u1@ s1@v1).transpose(-3, -1)
        
        size = [ y3.shape[-3] , y1.shape[-2] , y1.shape[-1]]
        y3 = F.interpolate(y3,size=size)
        y2 = F.interpolate(y2,size=size)
        y1 = F.interpolate(y1,size=size) 
        y = (y3 + y2 + y1)
        
#         b,c = s3.shape[:2]
        
#         y3 = s3 
#         y2 = s2.transpose(-3,-2)  
#         y1 = s1.transpose(-3,-1)  
         
        return y1 + y2 + y3

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
        
#         print(EN3.shape)
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


# 波掩码编码器
class WaveMaskEncoder(nn.Module):
    def __init__(self, fact=4, continuity=False, reshape=0) -> None:
        super().__init__()
        self.EN = False
        self.DE = False 
        self.fact = fact
        self.continuity = continuity
        self.reshape = reshape
        

    def make_en_de(self, y3d):
        x3 = F.adaptive_max_pool2d(y3d, 1).squeeze(-1).squeeze(-1) 
        x2 = F.adaptive_max_pool2d(y3d.transpose(-3, -2), 1).squeeze(-1).squeeze(-1) 
        x1 = F.adaptive_max_pool2d(y3d.transpose(-3, -1), 1).squeeze(-1).squeeze(-1) 
        
        EN3, DE3 = make_mask_wre(x3>0.6, self.fact, self.continuity)
        EN2, DE2 = make_mask_wre(x2>0.6, self.fact, self.continuity)
        EN1, DE1 = make_mask_wre(x1>0.6, self.fact , self.continuity)
        
#         print(EN3.shape)
        self.EN321 = [torch.FloatTensor(EN3 ).unsqueeze(0).unsqueeze(0).to(y3d.device),
                      torch.FloatTensor(EN2 ).unsqueeze(0).unsqueeze(0).to(y3d.device),
                      torch.FloatTensor(EN1 ).unsqueeze(0).unsqueeze(0).to(y3d.device)]
        self.DE321 = [torch.FloatTensor(DE3 ).unsqueeze(0).unsqueeze(0).to(y3d.device),
                      torch.FloatTensor(DE2 ).unsqueeze(0).unsqueeze(0).to(y3d.device),
                      torch.FloatTensor(DE1 ).unsqueeze(0).unsqueeze(0).to(y3d.device)]

        
    def encode(self, inx): 
        EN,DE = self.EN321,self.DE321
        x = (inx.transpose(-1,-3)@EN[0]).transpose(-1,-3)
        x = (x.transpose(-1,-2)@EN[1]).transpose(-1,-2)
        x1 = (x@EN[2]) 
        
        if self.reshape != 0:
            self.old_size = x1.shape[2:]
            s = self.reshape
            x1 = F.interpolate(x1, size=(s,s,s), mode='trilinear', align_corners=True) 
        
        return x1

    def decode(self, iny): 
        
        EN,DE = self.EN321,self.DE321
        
        if self.reshape != 0: 
            iny = F.interpolate(iny, size=self.old_size, mode='trilinear', align_corners=True)
            
#         print(iny.shape, DE[2].shape,EN[2].shape)
        y = (iny@DE[2]) 
        y = (y.transpose(-1,-2)@DE[1]).transpose(-1,-2)
        y1 = (y.transpose(-1,-3)@DE[0]).transpose(-1,-3)
        
         
        y = (iny.transpose(-1,-2)@DE[1]).transpose(-1,-2)
        y = (y.transpose(-1,-3)@DE[0]).transpose(-1,-3)
        y2 = (y@DE[2]) 
        
        
        y = (iny.transpose(-1,-3)@DE[0]).transpose(-1,-3)
        y = (y@DE[2]) 
        y3 = (y.transpose(-1,-2)@DE[1]).transpose(-1,-2)
        
        
        return (y1+y2+y3)/3

    
class RWAE(nn.Module):
    def __init__(self, c, q, gama, _lambda=0.3) -> None:
        super().__init__()
        self.swa = SvdWaveAttention(c, q)
        self.wre = WaveRandomEncoder(gama)
        self._lambda =  _lambda

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
        return  self._lambda*va + x


if __name__ == '__main__':
    b, c = 2, 2 # batchsize , channle
    d, w, h = 20, 20, 20
    gama = 0.1 # wre模块的特征保留系数
    q = 6 # 波特征数
    x = torch.randn((b, c, d, w, h)).cuda(0)  # batchsize*chanel, length, word
    rwae = RWAE(c,q,gama).cuda(0)
    rwae(x,LOG=True)
     
