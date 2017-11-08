#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MemoryAllocator.h"

#define tindex(thiz, d0, d1, d2, d3) \
        d0 * (thiz->D1 * thiz->D2 * thiz->D3) + \
        d1 * (thiz->D2 * thiz->D3) + \
        d2 * (thiz->D3) + \
        d3

#define sizeoftensor(thiz) \
        thiz->D0 * thiz->D1 * thiz->D2 * thiz->D3

/* Four dimension vectors */
typedef struct __tensor tensor;

struct __tensor {
    /* tensor[d0, d1, d2, d4] = value */
    void (*set)(tensor *thiz, int d0, int d1, int d2, int d3, int value);
    /* return the value by given position */
    int (*get)(tensor *thiz, int d0, int d1, int d2, int d3);
    /* copy data from host to device */
    void (*toGpu)(tensor *thiz);
    /* copy data from device to host */
    void (*toCpu)(tensor *thiz);
    /* Allocate memory on Host */
    void (*mallocHost)(tensor *thiz);
    /* Allocate memory on Device */
    void (*mallocDev)(tensor *thiz);

    /* info of this tensor */
    int D0, D1, D2, D3;
    /* address of data on Host */
    int *hostData;
    /* address of data on Device */
    int *devData;
};

void tensor_create(tensor **thiz, int d0, int d1, int d2, int d3);

#endif /* TENSOR_H_ */
