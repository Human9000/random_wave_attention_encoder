// cmd : gcc -fPIC -shared -o dll/wre_bnlved.dll cpp/wre_bnlved.cpp
extern "C" {
    void random_en_de(int B, int N, int L, bool *mask_vector, float *mae_en, float *mae_de);
    int bnl(int B, int N, int L, int b,  int n, int l);
    void mask_en_de(int N, int L, bool *MASK, float *EN, float *DE) ;
    int nl(int N, int L,  int n, int l);
}

#include<algorithm>
#include<ctime>
#include<stdio.h>
using namespace std;

int bnl(int B, int N, int L, int b,  int n, int l) {
    if (b>=B | n>=N | l>=L | b<0 | n<0 | l<0) {
        printf("Error(B,N,L,b,n,l):(%d,%d,%d,%d,%d,%d)\n",B,N,L,b,n,l);
    }
    return b*(N*L) + n*L + l;
} 

void random_en_de(int B, int N, int L, bool *mask_vector, float *mae_en, float *mae_de) {
    // batch_size,select number, length
    int l = 0, n = 0, b = 0;
    int  p = 0, q = L - 1; // 前项指针，后向指针
    float sum = 0; 
    srand ( unsigned ( time(0) ) ); // 设置随机时间戳

    for(b=0; b<B; b++){ 
        // 设置mask_vector值
        for (l=0; l<L; l++){
            if(l<N-1 | l==L-1) mask_vector[b*L + l] = true;
            else mask_vector[b*L + l] = false; 
        } 
        random_shuffle(mask_vector+1+b*L,mask_vector+(b+1)*L - 1); // 除头尾随机打乱
         
        //处理开头，跳过开头的非选中特征
        for (l = 0; l < L; l++)
            if (mask_vector[b*L + l] == true) {
                mae_de[bnl(B,N,L, b, 0, l)] = 1.0;
                mae_en[bnl(B,L,N, b, l, 0)] = 1.0;
                p = l; break;
            }
        //处理结尾，跳过结尾的非选中特征
        for (l = L - 1; l > -1; l--)
            if (mask_vector[b*L + l] == true) {
                mae_de[bnl(B,N,L, b, N-1, l)] = 1.0;
                mae_en[bnl(B,L,N, b, l, N-1)] = 1.0;
                q = l; break;
            }
        // 处理中间，正向处理遇到选中特征后，刷新递减衰弱的距离系数
        for (l= p + 1; l< L ; l++)
            if (mask_vector[b*L + l] == true) {
                if (++n == N) { n-=1; break; }
                mae_de[bnl(B,N,L, b, n, l)] = 1.0;
                mae_en[bnl(B,L,N, b, l, n)] = 1.0; 
                p = l;  
            } else if(n<N-1){
                mae_de[bnl(B,N,L, b, n, l)] = 1.0 / (l- p);
            }else if(n<N){
                mae_de[bnl(B,N,L, b, n, l)] = 1.0;
            }
        
        // 处理中间，反向处理，遇到选中特征后，刷新递减衰弱的距离系数
        for (l= q - 1; l> -1; l--)
            if (mask_vector[b*L + l] == true) {
                q = l; if (--n == -1) {  break; }
            } else if (n > 1) {
                sum = mae_de[bnl(B,N,L, b, n-1, l)] + 1.0 / (q - l);
                mae_de[bnl(B,N,L, b, n-1, l)] /= sum;
                mae_de[bnl(B,N,L, b, n, l)] = 1.0 / (q - l) / sum;
            } else if (n > 0) {
                mae_de[bnl(B,N,L, b, n, l)] = 1.0;
            }
    }
    
};


int nl(int N, int L,  int n, int l) {
    if ( n>=N | l>=L | n<0 | l<0) {
        printf("Error(N,L,n,l):(%d,%d,%d,%d)\n",N,L,n,l);
    }
    return n*L + l;
} 

void mask_en_de(int N, int L, bool *MASK, float *EN, float *DE) {
    // batch_size,select number, length
    int l = 0, n = 0, b = 0;
    int  p = 0, q = L - 1; // 前项指针，后向指针
    float sum = 0;  

    //处理开头，跳过开头的非选中特征
    for (l = 0; l < L; l++)
        if (MASK[l] == true) {
            DE[nl(N,L,  0, l)] = 1.0;
            EN[nl(L,N,  l, 0)] = 1.0;
            p = l; break;
        }
    //处理结尾，跳过结尾的非选中特征
    for (l = L - 1; l > -1; l--)
        if (MASK[l] == true) {
            DE[nl(N,L,  N-1, l)] = 1.0;
            EN[nl(L,N, l, N-1)] = 1.0;
            q = l; break;
        }
    // 处理中间，正向处理遇到选中特征后，刷新递减衰弱的距离系数
    for (l= p + 1; l< L ; l++)
        if (MASK[l] == true) {
            if (++n == N) { n-=1; break; }
            DE[nl(N,L, n, l)] = 1.0;
            EN[nl(L,N, l, n)] = 1.0; 
            p = l;  
        } else if(n<N-1){
            DE[nl(N,L, n, l)] = 1.0 / (l- p);
        }else if(n<N){
            DE[nl(N,L, n, l)] = 1.0;
        }

    // 处理中间，反向处理，遇到选中特征后，刷新递减衰弱的距离系数
    for (l= q - 1; l> -1; l--)
        if (MASK[l] == true) {
            q = l; if (--n == -1) {  break; }
        } else if (n > 1) {
            sum = DE[nl(N,L, n-1, l)] + 1.0 / (q - l);
            DE[nl(N,L, n-1, l)] /= sum;
            DE[nl(N,L, n, l)] = 1.0 / (q - l) / sum;
        } else if (n > 0) {
            DE[nl(N,L, n, l)] = 1.0;
        }
};
