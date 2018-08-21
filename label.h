#ifndef __MNIST_LABEL_H__
#define __MNIST_LABEL_H__

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/client/client_session.h"

#include "utils.h"

#define LABEL_MAGIC_NUMBER 2049

class Label 
{
private: 
    std::string uri;
    char *data;
    int mn, numLabels;
    int batchStartIndex;

public:
    Label(std::string labelUri);
    Label();
    ~Label();
    tensorflow::Tensor *tensor;
    tensorflow::Tensor *batchTensor;

    void loadBatch(int batchStartIndex, int batchSize);
    void read(void);
};

#endif

