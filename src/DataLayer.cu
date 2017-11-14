#include "tensor.h"
#include "DataLayer.h"

static void feedforward(LayerBase *thiz)
{
    DataLayer *data = (DataLayer *) thiz;
    /* data flowing from input to output */
    data->lb->output = data->lb->input;
    /* prepare output on GPU memory */
    data->lb->output->mallocDev(data->lb->output);
    data->lb->output->toGpu(data->lb->output);
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
    base->output = NULL;
    base->preLayer = NULL;
    base->feedforward = feedforward;
    /* Unkown Next Layer */
    base->nextLayer = NULL;
}
