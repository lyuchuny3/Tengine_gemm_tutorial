/*
 * Author: Chunying
 * 2019/08/27
 */

#include "gemm_utils.h"
#include<cblas.h>
#include<sys/time.h>
#include<stdlib.h> //setenv

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
void gemm_blas(float* A,float* B,float* C,int m,int n,int k)
{
    // C=alpha*A*B+beta*C 
    int alpha =1;
    int beta = 0;
    cblas_sgemm(CblasRowMajor, 
                CblasNoTrans, CblasNoTrans, 
                m, n, k, 
                alpha, 
                A, k,
                B, n, 
                beta, 
                C, n);
}
int main(int argc,char** argv)
{
    int m=256;
    int n=128;
    int k=256;
    printf("[m n k]:\t%d %d %d\n",m,n,k);

    float* A    =  init(m*k,4);
    float* B    =  init(n*k,4);
    float* C1    =  init(m*n,0);
    float* C2    =  init(m*n,0);
    
    int rep = 50;
    struct timeval t0, t1;
    //blas
    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        gemm_blas(A,B,C1,m,n,k);
    gettimeofday(&t1, NULL);
    float blas_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[openblas]:\t%.2f ms\n", blas_time / rep);

    //pure c
    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        gemm_pure_c(A,B,C2,m,n,k);
    gettimeofday(&t1, NULL);
    float c_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[pure c]:\t%.2f ms\n", c_time / rep);

    printf("[blas VS pure_C]:  maxerr=%f \n",maxerr(C1,C2,n*m));

    free(A);
    free(B);
    free(C1);
    free(C2);
    return 0;
}