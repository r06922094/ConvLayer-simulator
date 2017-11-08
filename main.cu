#include <stdio.h>
#include <stdlib.h>
#include "src/tensor.h"
#include "src/MemoryAllocator.h"
#include "src/net.h"

int main()
{
    MemoryAllocator_create(&ma);
    
    tensor *test;
    tensor_create(&test, 100, _INPUT_DIM, _INPUT_DIM, _IMAGE_CHANNEL);
  
    LayerBase *head = buildNetwork();
    trainNetwork(head, test);
    

    return 0;
}
