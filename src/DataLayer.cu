#include "tensor.h"
#include "DataLayer.h"

static void feedforward(LayerBase *thiz)
{
    DataLayer *data = (DataLayer *) thiz;
    tensor *output = data->lb->output;
    /* Prepare a new copy for flowing data */
    //tensor_create(&output, base->batchSize, base->inputDim, base->inputDim, base->inputChannel);
    /* TODO: Copy data from input to output */
    output->toGpu(output);
}

/* DataLayer has no preLayer */
void DataLayer_init(DataLayer **thiz, int batchSize, \
                    int inputDim, \
                    int inputChannel, \
                    LayerBase *nextLayer)
{
    (*thiz) = (DataLayer *) malloc(sizeof(DataLayer));
    if(!(*thiz)) {
        printf("DataLayer.c: No memory available\n");
        exit(0);
    }
    (*thiz)->lb = (LayerBase *) malloc(sizeof(LayerBase));
    if(!(*thiz)->lb) {
        printf("DataLayer.c: No memory available\n");
        exit(0);
    }

    LayerBase *base = (*thiz)->lb;
    base->batchSize = batchSize;
    base->inputDim = inputDim;
    base->outputDim = inputDim;
    base->inputChannel = inputChannel;
    base->outputChannel = inputChannel;
    base->input = NULL;
    /* Allocate memory for output */
    tensor_create(&base->output, base->batchSize, base->outputDim, base->outputDim, base->outputChannel);
    /******************************/
    base->preLayer = NULL;
    base->feedforward = feedforward;
    /* TODO: unknown nextLayer */
    base->nextLayer = NULL;
}
