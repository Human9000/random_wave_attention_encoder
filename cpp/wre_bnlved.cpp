// cmd : gcc -fPIC -shared -o dll/wre_bnlved.dll cpp/wre_bnlved.cpp
extern "C" {
    void random_en_de(int B, int N, int L, bool *mask_vector, float *mae_en, float *mae_de);
    int bnl(int B, int N, int L, int b,  int n, int l);
}

#include<algorithm>
#include<ctime>
using namespace std;

int bnl(int B, int N, int L, int b,  int n, int l) {
    return b*(N*L) + n*L + l;
}
 
void random_en_de(int B, int N, int L, bool *mask_vector, float *mae_en, float *mae_de) {
    // batch_size,select number, length
    int l = 0, n = 0, b = 0;
    int  p = 0, q = L - 1; // 前项指针，后向指针
    float sum = 0;
    // mask_vector; // 设置时间戳
    srand ( unsigned ( time(0) ) );

    for(b=0; b<B; b++){
        random_shuffle(mask_vector+1+b*N*L,mask_vector+(b+1)*N*L - 1); // 除头尾随机打乱

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
                mae_de[bnl(B,N,L, b, ++n, l)] = 1.0;
                mae_en[bnl(B,L,N, b, l, n)] = 1.0;
                // printf("%d %d %d\n",b,n,l);
                p = l;  if (n == N) { n-=1; break; }
            } else if(l<q){
                mae_de[bnl(B,N,L, b, n, l)] = 1.0 / (l- p);
            }else{
                mae_de[bnl(B,N,L, b, n, l)] = 1.0;
            }
        
        // 处理中间，反向处理，遇到选中特征后，刷新递减衰弱的距离系数
        for (l= q - 1; l> -1; l--)
            if (mask_vector[b*L + l] == true) {
                --n; q = l; if (n == -1) {  break; }
            } else if (n > 0) {
                sum = mae_de[bnl(B,N,L, b, n-1, l)] + 1.0 / (q - l);
                mae_de[bnl(B,N,L, b, n-1, l)] /= sum;
                mae_de[bnl(B,N,L, b, n, l)] = 1.0 / (q - l) / sum;
            } else {
                mae_de[bnl(B,N,L, b, n, l)] = 1.0;
            }
    }
    
};
