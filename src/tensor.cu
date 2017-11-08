#include "tensor.h"

static void toGpu(tensor *thiz)
{
    /* Allocate Memory for Gpu */
    cudaError_t error;
    int size = sizeoftensor(thiz);

    if (thiz->devData) {
        error = cudaMemcpy(thiz->devData, thiz->hostData, sizeof(int) * size, cudaMemcpyHostToDevice);
    } else {
        thiz->mallocDev(thiz);
        error = cudaMemcpy(thiz->devData, thiz->hostData, sizeof(int) * size, cudaMemcpyHostToDevice);
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
        error = cudaMemcpy(thiz->hostData, thiz->devData, sizeof(int) * size, cudaMemcpyDeviceToHost);
    } else {
        printf("tensor.c: toCpu failed\n");
        exit(0);
    }

    if (error != cudaSuccess) {
        printf("tensor.c: toCpu failed\n");
        exit(0);
    }
}

static void set(tensor *thiz, int d0, int d1, int d2, int d3, int value)
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
static int get(tensor *thiz, int d0, int d1, int d2, int d3)
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

    thiz->hostData = (int *) ma->HostMalloc(ma, sizeof(int) * size);
    memset(thiz->hostData, 0, thiz->D0 * thiz->D1 * thiz->D2 * thiz->D3 * sizeof(int));
}

static void mallocDev(tensor *thiz)
{
    int size = sizeoftensor(thiz);
    if (thiz->devData) {
        printf("tensor.c: Repeatly allocate gpu memory for tensor\n");
        exit(0);
    }

    thiz->devData = (int *) ma->DevMalloc(ma, sizeof(int) * size);
    cudaError_t error = cudaMemset(thiz->devData, 0, sizeof(int) * size);
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
