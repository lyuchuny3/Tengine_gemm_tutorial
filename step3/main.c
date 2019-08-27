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
// tengine 1
extern void tengine_4x16_kernel(float* C, float* B,float* A,int k);
void interleave_A16(float* src,float* dst,int m,int k)
{
    float* ptr = dst;
    for(int i=0;i< m;i+=16)
    {
        for(int j=0;j<k;j++)
        {
            for(int p=0;p<16;p++)
            {
                *ptr = src[(i+p)*k +j];
                ptr++;
            }
        }
    }
}
void interleave_B4(float* src,float* dst,int n,int k)
{
    float* ptr = dst;
    for(int i=0;i< n;i+=4)
    {
        for(int j=0;j<k;j++)
        {
            for(int p=0;p<4;p++)
            {
                *ptr = src[j*n+ i+p];
                ptr++;
            }
        }
    }
}
void sgemm_A16_B4(float* mid_A,float* B,float* mid_B,float* C,
        int m,int n,int k)
{
    interleave_B4(B,mid_B,n,k);
    float result[64];
    for(int i=0;i<m;i+=16)
    {
        for(int j=0;j<n;j+=4)
        {
            tengine_4x16_kernel(result, 
                            mid_B + j*k ,
                            mid_A + i*k,
                            k);
            for(int p = 0; p < 16; p++)
            {
                for(int q = 0; q < 4; q++)
                {
                    *(C + (i + p) * n + j + q) = result[(p << 2) + q];
                }
            }
        }
    }
}

int main(int argc,char** argv)
{
    int m=256;
    int n=256;
    int k=256;

    printf("[m n k]:\t%d %d %d\n",m,n,k);
    if((m%16!=0) ||(n%4!=0)||(k%4!=0))
    {
        printf("This demo only support:\n m a multiple of 16\tn a multiple of 4\tk a multiple of 4\n");
        printf(" please verify your m,n,k\n");
        return -1;
    }

    float* A    = init(m*k,4);
    float* B    = init(n*k,4);
    float* C1   = init(m*n,0);
    float* C2   = init(m*n,0);
    float* C3   = init(m*n,0);
    float* midA = init(m*k,0);
    float* midB = init(n*k,0);

    int rep = 50;
    struct timeval t0, t1;

    //tengine 4x16
    interleave_A16(A,midA,m,k);
    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        sgemm_A16_B4(midA,B,midB,C1,m,n,k);
    gettimeofday(&t1, NULL);
    float t_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[tengine 4x16]:\t%.2f ms\n", t_time / rep);

    //blas
    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        gemm_blas(A,B,C2,m,n,k);
    gettimeofday(&t1, NULL);
    float blas_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[openblas]:\t%.2f ms\n", blas_time / rep);

    //pure c
    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        gemm_pure_c(A,B,C3,m,n,k);
    gettimeofday(&t1, NULL);
    float c_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[pure c]:\t%.2f ms\n", c_time / rep);

    //maxerr
    printf("[blas VS tengine]:  maxerr=%f \n",maxerr(C1,C2,n*m));

    //free
    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    free(midA);
    free(midB);

    return 0;
}