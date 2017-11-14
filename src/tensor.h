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

/* Unit for Neural Network to operate */
typedef float conv_unit_t;

struct __tensor {
    /* tensor[d0, d1, d2, d4] = value */
    void (*set)(tensor *thiz, int d0, int d1, int d2, int d3, conv_unit_t value);
    /* return the value by given position */
    conv_unit_t (*get)(tensor *thiz, int d0, int d1, int d2, int d3);
    /* copy data from host to device */
    void (*toGpu)(tensor *thiz);
    /* copy data from device to host */
    void (*toCpu)(tensor *thiz);
    /* Allocate memory on Host */
    void (*mallocHost)(tensor *thiz);
    /* Allocate memory on Device */
    void (*mallocDev)(tensor *thiz);
    /* Return Host address */
    conv_unit_t *(*getHost)(tensor *thiz);
    /* Return Device address */
    conv_unit_t *(*getDev)(tensor *thiz);

    /* info of this tensor */
    int D0, D1, D2, D3;
    /* address of data on Host */
    conv_unit_t *hostData;
    /* address of data on Device */
    conv_unit_t *devData;
};

void tensor_create(tensor **thiz, int d0, int d1, int d2, int d3);

#endif /* TENSOR_H_ */
