[中文版本](readme_cn.md)

# Tengine Gemm Tutorials On ARM-v8
@author: Chunying

## GEMM Introduction
What is GEMM? It stands for GEneral Matrix to Matrix Multiplication. Gemm is an important part of neural network computations. Here is  the article [Why gemm is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/) that explains why GEMM is important for deep learning and how GEMM works for Convolutions.

## Outline
In this tutorial, we focus on how to optimize GEMM on ARM-V8. There are mainly three parts:
- [Step1: GEMM with Pure C Code](#step1-gemm-with-pure-c-code)

    This part introduces the basic notations of gemm (A,B,C,m,n,k).
- [Step2: GEMM with OpenBLAS](#step2-gemm-with-openblas)
  
    This part uses `cblas_sgemm` of the Openblas library, and compare the performance between blas and pure c implementation.
- [Step3: GEMM with Tengine_4x16_kernel](#step3-gemm-with-tengine-16x4-kernel)
    
    This part takes [Tengine](https://github.com/OAID/Tengine)'s source code [sgemm_4x16_interleave.S](https://github.com/OAID/Tengine/blob/master/executor/operator/arm64/conv/sgemm_4x16_interleave.S) as example, explains how the data interleave and shows the performance compared to OpenBLAS.

To run the codes in this tutorial, you need:
- some boards that support armv8 assembly, like **RK3399**
- Linux OS: this tutorial use Makefile (if on android, you can write your own CmakeLists.txt)
## Step1: GEMM with Pure C Code
First, we run the codes to see the outputs, and then we explain the implementations.
```bash
cd step1
make
./test
```
You should get:
```c
A=
3.000000 2.000000 1.000000 3.000000
1.000000 3.000000 2.000000 0.000000
1.000000 1.000000 2.000000 3.000000
2.000000 3.000000 3.000000 2.000000
================
B=
3.000000 2.000000 1.000000 3.000000
1.000000 3.000000 2.000000 0.000000
1.000000 1.000000 2.000000 3.000000
2.000000 3.000000 3.000000 2.000000
================
C=
18.000000 22.000000 18.000000 18.000000
8.000000 13.000000 11.000000 9.000000
12.000000 16.000000 16.000000 15.000000
16.000000 22.000000 20.000000 19.000000
================
```
We compute A(m,k) * B(k,n), and get C(m,n).
* A is the 1st input matrix.
  * m is the number of rows of A
  * k is the number of columns of A
* B is the 2nd input matrix
  * k is the number of rows of B
  * n is the number of columns of B
* C is the output matrix
  * m is the number of rows of C
  * n is the number of columns of C
  
![image](imgs/gemm.jpg)

We initialize A,B,C using `init` function in `gemm_utils.h`
```c
    float* A    =  init(m*k,3);
    float* B    =  init(n*k,3);
    float* C    =  init(m*n,0);
```
Then we implement the gemm with pure c code
```c
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
```
calling this function
```c
    gemm_pure_c(A,B,C,m,n,k);
```
and we print out the three matrix A,B,C. 
```
    printf("A=\n");printM(A,m,k);
    printf("B=\n");printM(B,k,n);
    printf("C=\n");printM(C,m,n);
```
You can compute this matrix multiplication `by hand`, and verify the result with the output of the program.

## Step2: GEMM with OpenBLAS
### What is OpenBLAS?

[OpenBLAS](https://www.openblas.net/) is an open source implementation of the BLAS (Basic Linear Algebra Subprograms) API with many hand-crafted optimizations for specific processor types. 

### How to Install OpenBLAS?

On Linux, you can install by:
> sudo apt-get install libopenblas-dev

### How to Run?
First, we run the codes to see the outputs, and then we explain the implementations.
```bash
cd step2

taskset 0x1 ./test
```
You should get:
```
[m n k]:        256 128 256
[openblas]:     4.68 ms
[pure c]:       32.22 ms
[blas VS pure_C]:  maxerr=0.000076
```

Here we focus the performance on **one single CPU**. On RK3399, we use 1A53. 
* We set the number of threads of OMP to 1 by 
    > export OMP_NUM_THREADS=1
* We then bind the test program on one cpu by `taskset`
    > taskset 0x1 ./test

We can see the performance of OpenBLAS is obviously better than the pure C codes. We also compute the max-error of these two implementations inorder to verify the correctness of the results.

We call the `cblas_sgemm` function of the OpenBLAS library, remember to add the include file `<cblas.h>`,  and add `-lopenblas` in Makefile
```c
#include <cblas.h>
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
```
For timing, we repeat 50 times and get the average time.
```c
    int rep = 50;
    struct timeval t0, t1;

    gettimeofday(&t0, NULL);
    for(int i = 0; i < rep; i++)
        gemm_blas(A,B,C1,m,n,k);
    gettimeofday(&t1, NULL);
    
    float blas_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    printf("[openblas]:\t%.2f ms\n", blas_time / rep);
```
## Step3: GEMM with Tengine 16x4 kernel
First, we run the codes to see the performance on RK3399 1A53, and then we explain the implementations.
```bash
cd step3
make
export OMP_NUM_THREADS=1
taskset 0x1 ./test
```
You can get the outputs as:
```
[m n k]:        256 256 256
[tengine 4x16]: 7.71 ms
[openblas]:     9.55 ms
[pure c]:       316.00 ms
[blas VS tengine]:  maxerr=0.000061
```
We can see the Tengine 4x16 kernel gets the best performance of the three implementations. Well, how this Tengine 4x16 kernel works?

This part takes [Tengine](https://github.com/OAID/Tengine)'s source code [sgemm_4x16_interleave.S](https://github.com/OAID/Tengine/blob/master/executor/operator/arm64/conv/sgemm_4x16_interleave.S) as example. We simplify this assembly file, only support for k is a multiple of 4.

**Interleave**

Before using the 4x16 kernel, we firstly interleave matrix A and matrix B. So what does interleave means? Interleaving means arranging data in some manners for the sake of cache efficiency.

For Tengine 4x16 kernel, we interleave matrix A for every 16 elements of m, and for matrix B every 4 elements of n. 

```c
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
```
![image](imgs/interleave_B.gif)

```c
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
```

**tengine_4x16_kernel**

The Tengine 4x16 kernel compute A(16,k)*B(k,4) = C(16,4).

we compute in `loop4` every 4 elements in k. 
* load data of B using register `v0,v1,v2,v3`
* load data of A `v4,v5,v6,v7,v8,v9,v10,v11`

```ASM
	//load data of B
	ldr	q0, [x1]	
	ldr	q1, [x1, 0x10]	
	ldp	q2, q3, [x1, 0x20]

	//load data of A
	ldp	q4, q5, [x2]
	ldp	q6, q7, [x2, 0x20]		
	ldp	q8, q9, [x2, 0x40]
	ldp	q10,q11,[x2, 0x60]
```
The following gif shows how each instruction compute in kernel 4x16

![image](imgs/gemm_16x4.gif)

Finally, store the output
```
	stp     q16, q17 ,[x0]
	stp     q18, q19 ,[x0, #0x20]
	stp     q20, q21 ,[x0, #0x40]
	stp     q22, q23 ,[x0, #0x60]
	stp     q24, q25 ,[x0, #0x80]
	stp     q26, q27 ,[x0, #0xa0]
	stp     q28, q29 ,[x0, #0xc0]
	stp     q30, q31 ,[x0, #0xe0]
```
and arrange the output
```c
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
```

## What's More?
This tutorial is only an exercise. The codes in part3 only supports:
- [x] m as a multiple of 16
- [x] n as a multiple of 4
- [x] k as a multiple of 4
  
There're lots of works you can do after this tutorial:
* you can extend the codes to support random k. Ref  [sgemm_4x16_interleave.S](https://github.com/OAID/Tengine/blob/master/executor/operator/arm64/conv/sgemm_4x16_interleave.S) add `loop1`.
* you can translate the function of `interleave_B4` into assembly code for better performance.
* you can extend the codes to support random `m` and `n`, by padding into multiple of 4/16.
* you can try your own `4x4_kernel.S` for armv8
* you can try to write `4x4_kernel.S` for armv7
  

## Link
- Tengine doc:https://tengine-docs.readthedocs.io/zh_CN/latest/index.html
