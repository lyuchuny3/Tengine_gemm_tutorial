/*
 * Author: Chunying
 * 2019/08/27
 */

#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

static inline void DumpFloat(const char *fname, float *data, int number)
{
    FILE *fp = fopen(fname, "w");

    for (int i = 0; i < number; i++) {
        if (i % 8 == 0) {
            fprintf(fp, "\n%d:",i);
        }
        fprintf(fp, " %.5f", data[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

float maxerr(float *gt, float *pred, int size)
{
    float maxError = 0.f;
    float tmp =0.f;
    for (int i = 0; i < size; i++) {
        tmp = (float)fabs(gt[i] - pred[i]);
        if (tmp > 0.01) {
            printf("==============================================\n");
            printf("mismatch at idx=%d, pred=%f, gt=%f\n", i, pred[i], gt[i]);
            printf("dump data to file [gt_data, pred_data]\n");
            printf("=============================================\n");
            DumpFloat("gt_data", gt, size);
            DumpFloat("pred_data", pred, size);
            return -1;
        }
        maxError = MAX(tmp, maxError);
    }
    return maxError;
}

static inline float *init(int size, int mode)
{
    srand(0); //set rand_seed
    int i;
    float *m = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; ++i) {
        if (mode == 0)
            m[i] = 0;
        else if (mode == 1)
            m[i] = 1;
        else if (mode == 2)
            m[i] = i % 8;
        else if (mode == 3)
            m[i] = (float)(rand()%4); 
        else
            m[i] = (float)rand() / RAND_MAX;
    }
    return m;
}
void printM(float* x,int h,int w)
{
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            printf("%f ",x[i*w+j]);
        }
        printf("\n");
    }
    printf("================\n");
}
#endif //GEMM_UTILS_H