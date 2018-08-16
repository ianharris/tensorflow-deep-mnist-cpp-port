#ifndef __MNIST_IMAGE_H__
#define __MNIST_IMAGE_H__

#include <string>

#include "tensorflow/core/framework/tensor.h"

#include "utils.h"

#define IMAGE_MAGIC_NUMBER 2051

class Image {
private:
    std::string uri;
    uint8_t *data;
    int mn, numImages, rows, columns;

public:
    // constructors / destructors
    Image(std::string imageUri);
    Image();
    ~Image();

    // tensor
    tensorflow::Tensor *tensor;
    tensorflow::Tensor *batchTensor;

    // load batch
    void loadBatch(int batchStartIndex, int batchSize);

    // read images from disk
    void read(void);

    // getters
    int getNumImages(void);
    int getNumRows(void);
    int getNumColumns(void);
};

#endif

