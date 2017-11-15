CC := /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -O4 -g -gencode arch=compute_61,code=sm_61 --ptxas-options=-v \
         -I/usr/local/cuda/samples/common/inc/ \
		 -I/usr/local/include/opencv/ \
		 -I/usr/local/include/opencv2/ \
		 -L/usr/local/lib/
LINK := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -l opencv_highgui
OUT ?= ./build
SRC ?= ./src


$(OUT)/tensor.o: $(SRC)/tensor.cu $(SRC)/tensor.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/MemoryAllocator.o: $(SRC)/MemoryAllocator.cu $(SRC)/MemoryAllocator.h $(OUT)
	$(CC) $(CLFAGS) -c -o $@ $<

$(OUT)/DataLayer.o: $(SRC)/DataLayer.cu $(SRC)/DataLayer.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/ConvLayer.o: $(SRC)/ConvLayer.cu $(SRC)/ConvLayer.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/net.o: $(SRC)/net.cu $(SRC)/net.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

main: main.cu $(OUT)/tensor.o $(OUT)/MemoryAllocator.o \
	  $(OUT)/DataLayer.o $(OUT)/ConvLayer.o \
	  $(OUT)/net.o
	$(CC) $(CFLAGS) -o $@ $^ $(LINK)

clean:
	rm $(OUT)/*.o main
