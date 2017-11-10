#ifndef MEMORY_ALLOCATOR_H_
#define MEMORY_ALLOCATOR_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct __MemoryAllocator MemoryAllocator;

struct __MemoryAllocator {
    /* Malloc memory on Host */
    void *(*HostMalloc)(MemoryAllocator *thiz, int size);
    /* Malloc memory on Device */
    void *(*DevMalloc)(MemoryAllocator *thiz, int size);
    
    /* TODO: Print and free memory */

    int HostMemory;
    int DevMemory;
};

void MemoryAllocator_create(MemoryAllocator **thiz);

extern MemoryAllocator *ma;

#endif
