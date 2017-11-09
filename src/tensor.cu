#include "tensor.h"

static void toGpu(tensor *thiz)
{
    /* Allocate Memory for Gpu */
    cudaError_t error;
    int size = sizeoftensor(thiz);

    if (thiz->devData) {
        error = cudaMemcpy(thiz->devData, thiz->hostData, sizeof(conv_unit_t) * size, cudaMemcpyHostToDevice);
    } else {
        thiz->mallocDev(thiz);
        error = cudaMemcpy(thiz->devData, thiz->hostData, sizeof(conv_unit_t) * size, cudaMemcpyHostToDevice);
    }

    if (error != cudaSuccess) {
        printf("tensor.c: toGpu failed\n");
        exit(0);
    }
}

static void toCpu(tensor *thiz)
{
    /* Check whether Gpu is available to copy */
    cudaError_t error;
    int size = sizeoftensor(thiz);
    if (thiz->devData) {
        error = cudaMemcpy(thiz->hostData, thiz->devData, sizeof(conv_unit_t) * size, cudaMemcpyDeviceToHost);
    } else {
        printf("tensor.c: toCpu failed\n");
        exit(0);
    }

    if (error != cudaSuccess) {
        printf("tensor.c: toCpu failed\n");
        exit(0);
    }
}

static void set(tensor *thiz, int d0, int d1, int d2, int d3, conv_unit_t value)
{
    int index = tindex(thiz, d0, d1, d2, d3);
    
    if ((d0 < 0 || d0 >= thiz->D0) || (d1 < 0 || d1 >= thiz->D1) || \
        (d2 < 0 || d2 >= thiz->D2) || (d3 < 0 || d3 >= thiz->D3)) {
        printf("tensor.c: index of tensor to access is out of range\n");
        exit(0);
    }
    
    thiz->hostData[index] = value;
}

/* return the value by given position */
static conv_unit_t get(tensor *thiz, int d0, int d1, int d2, int d3)
{
    int index = tindex(thiz, d0, d1, d2, d3);
    return thiz->hostData[index];
}

static void mallocHost(tensor *thiz)
{
    int size = sizeoftensor(thiz);
    
    if (thiz->hostData) {
        printf("tensor.c: Repeatly allocate host memory for tensor\n");
        exit(0);
    }

    thiz->hostData = (conv_unit_t *) ma->HostMalloc(ma, sizeof(conv_unit_t) * size);
    memset(thiz->hostData, 0, thiz->D0 * thiz->D1 * thiz->D2 * thiz->D3 * sizeof(conv_unit_t));
}

static void mallocDev(tensor *thiz)
{
    int size = sizeoftensor(thiz);
    if (thiz->devData) {
        printf("tensor.c: Repeatly allocate gpu memory for tensor\n");
        exit(0);
    }

    thiz->devData = (conv_unit_t *) ma->DevMalloc(ma, sizeof(conv_unit_t) * size);
    cudaError_t error = cudaMemset(thiz->devData, 0, sizeof(conv_unit_t) * size);
    if (error != cudaSuccess) {
        printf("tensor.c: cudamemset failed\n");
        exit(0);
    }
}

void tensor_create(tensor **thiz, int d0, int d1, int d2, int d3)
{
    (*thiz) = (tensor *) malloc(sizeof(tensor));
    (*thiz)->set = set; 
    (*thiz)->get = get;
    (*thiz)->toGpu = toGpu;
    (*thiz)->toCpu = toCpu;
    (*thiz)->mallocHost = mallocHost;
    (*thiz)->mallocDev = mallocDev;
    (*thiz)->D0 = d0;
    (*thiz)->D1 = d1;
    (*thiz)->D2 = d2;
    (*thiz)->D3 = d3;
    (*thiz)->hostData = NULL;
    (*thiz)->devData = NULL;
    /* Allocate Host memory when initialization */
    mallocHost(*thiz);
}
