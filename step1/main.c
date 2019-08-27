/*
 * Author: Chunying
 * 2019/08/27
 */

#include "gemm_utils.h"

void gemm_pure_c(float* A, float* B, float* C,int m,int n,int k)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            C[i*n+j]=0.f;
            for(int p=0;p<k;p++)
            {
                C[i*n+j]+=A[i*k+p]*B[p*n+j];
            }
        }
    }
}
int main(int argc,char** argv)
{
    int m=4;
    int n=4;
    int k=4;
    float* A    =  init(m*k,3);
    float* B    =  init(n*k,3);
    float* C    =  init(m*n,0);

    gemm_pure_c(A,B,C,m,n,k);

    printf("A=\n");printM(A,m,k);
    printf("B=\n");printM(B,k,n);
    printf("C=\n");printM(C,m,n);

    free(A);
    free(B);
    free(C);
    return 0;
}