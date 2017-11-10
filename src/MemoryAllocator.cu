#include "MemoryAllocator.h"

static void *HostMalloc(MemoryAllocator *thiz, int size)
{
    void *mem = NULL;
    //cudaError_t error = cudaHostAlloc(&mem, size, cudaHostAllocDefault);
    //if (error != cudaSuccess) {
    //    printf("MemoryAllocator.c: No available memory for Host: %d\n", size);
    //    exit(0);
    //}
    mem = malloc(size);
    if (!mem) {
        printf("MemoryAllocator.c: No available memory for Host: %d\n", size);
        exit(0);
    }
    /* Record memory usage of Host */
    thiz->HostMemory += size;

    return mem;
}

static void *DevMalloc(MemoryAllocator *thiz, int size)
{
    void *mem = NULL;
    cudaError_t error = cudaMalloc(&mem, size);
    if (error != cudaSuccess) {
        printf("MemoryAllocator.c: No available memory for Device: %d\n", error);
        exit(0);
    }
    /* Record memory usage of Device */
    thiz->DevMemory += size;

    return mem;
}

void MemoryAllocator_create(MemoryAllocator **thiz)
{
    (*thiz) = (MemoryAllocator *) malloc(sizeof(MemoryAllocator));
    (*thiz)->HostMemory = 0;
    (*thiz)->DevMemory = 0;
    (*thiz)->HostMalloc = HostMalloc;
    (*thiz)->DevMalloc = DevMalloc;
}

MemoryAllocator *ma;
