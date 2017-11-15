#include <stdio.h>
#include <stdlib.h>
#include "src/tensor.h"
#include "src/MemoryAllocator.h"
#include "src/net.h"
#include "highgui/highgui_c.h"
#include "imgcodecs/imgcodecs_c.h"

#define FLOAT(x) (x * 100.0 / 100.0)

int main(int argc, char *argv[])
{
    MemoryAllocator_create(&ma);
    
    tensor *test;
    
    IplImage *img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);//CV_LOAD_IMAGE_COLOR
    if (!img) {
        printf("No availabe image\b");
        exit(0);
    }
    //tensor_create(&test, 1, img->width, img->height, img->nChannels);
    int customChannel = atoi(argv[3]);
    tensor_create(&test, 1, img->width, img->height, customChannel);
    /* Set input value */
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < test->D1; j++) {
            for (int k = 0; k < test->D2; k++) {
                uchar *ptr = (uchar *) img->imageData + ( k + j * img->width) * img->nChannels;
                for (int w = 0; w < test->D3; w++) {
                    test->set(test, i, j, k, w, FLOAT(ptr[0]));
                }
            }
        }
    }

    LayerBase *head = buildNetwork(img->width, customChannel, atoi(argv[2]));
    tensor *result = trainNetwork(head, test);
    cudaDeviceSynchronize();
    printf("size: %d %d %d\n",result->D1, result->D2, result->D3);
    IplImage *output = cvCreateImage(cvSize(result->D1, result->D2), IPL_DEPTH_8U,  result->D3);

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < result->D1; j++) {
            for (int k = 0; k < result->D2; k++) {
                uchar *ptr = (uchar *)output->imageData + (( k + j * output->width) * output->nChannels);
                    for (int w = 0; w < result->D3; w++) {
                        ptr[w] = (uchar)(abs)(result->get(result, i, j, k, w));
                    }
            }
        }
    }

    cvSaveImage("./output.jpg", output, NULL);

    

    return 0;
}
