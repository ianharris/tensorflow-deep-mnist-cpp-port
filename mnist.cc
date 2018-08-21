#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <list>

// include tensorflow headers
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "image.h"
#include "label.h"

#define VERBOSE 0

void loadBatch(Image &trainImage, 
               Label &trainLabel)
{
    static int batchStartIndex = 0;
    int BATCH_SIZE = 50;

    trainImage.loadBatch(batchStartIndex, BATCH_SIZE);
    trainLabel.loadBatch(batchStartIndex, BATCH_SIZE);

    batchStartIndex += BATCH_SIZE;
    batchStartIndex %= trainImage.getNumImages();
}

tensorflow::ops::Variable weight_variable(tensorflow::Scope &root,
                                          tensorflow::ClientSession &session,
                                          std::initializer_list<tensorflow::int64> shape)
{
    if(VERBOSE) std::cout << "Creating the weight variable" << std::endl << std::flush;
    // create the variable
    auto w = tensorflow::ops::Variable(root, 
                                tensorflow::TensorShape(shape),
                                tensorflow::DT_DOUBLE);
    if(VERBOSE) std::cout << "Weight created" << std::endl << std::flush;
    // create the variables value
    auto wval = tensorflow::ops::TruncatedNormal(root, 
                                          tensorflow::Input(shape),
                                          tensorflow::DT_DOUBLE);
    if(VERBOSE) std::cout << "Weight val created" << std::endl << std::flush;
    // assign the variable to the value
    auto assign_w = tensorflow::ops::Assign(root,
                                                          w,
                                                          wval);
    if(VERBOSE) std::cout << "Weight assign prepared" << std::endl << std::flush;
    TF_CHECK_OK(session.Run({assign_w},
                            nullptr));
    if(VERBOSE) std::cout << "Weight assign run" << std::endl << std::flush;
    return w;
}

tensorflow::ops::Variable bias_variable(tensorflow::Scope &root,
                                        tensorflow::ClientSession &session,
                                        std::initializer_list<tensorflow::int64> shape)
{
    // create the bias variable
    if(VERBOSE) std::cout << "Creating the bias variable" << std::endl << std::flush;
    auto b = tensorflow::ops::Variable(root,
                                tensorflow::TensorShape(shape),
                                tensorflow::DT_DOUBLE);
    if(VERBOSE) std::cout << "Bias Variable created" << std::endl << std::flush;
    // create the variables value
    // tensorflow::ops::Fill bval(root,
    //                            tensorflow::Input(shape),
    //                            tensorflow::Input(0.1));
    auto bval = tensorflow::ops::TruncatedNormal(root, 
                                          tensorflow::Input(shape),
                                          tensorflow::DT_DOUBLE);
    if(VERBOSE) std::cout << "Bias variable value created" << std::endl << std::flush;

    // assign the value to the variable
    auto assign_b = tensorflow::ops::Assign(root,
                                     b,
                                     bval);
    assign_b.ValidateShape(true);
    if(VERBOSE) std::cout << "Bias value assignment created" << std::endl << std::flush;

    TF_CHECK_OK(session.Run({assign_b},
                            nullptr));
    if(VERBOSE) std::cout << "bias assignment run" << std::endl << std::flush;
    return b;
}

tensorflow::Output deepnn(tensorflow::Scope &root,
                          tensorflow::ClientSession &session,
                          tensorflow::ops::Placeholder &x)
{

    if(VERBOSE) std::cout << "Creating the first layer" << std::endl << std::flush;
    // create the first convolutional layer
    // - create the weight variable
    auto w1 = weight_variable(root,
                              session,
                              {5, 5, 1, 32});
    // - create the bias variable
    auto b1 = bias_variable(root,
                            session,
                            {32});
    
    if(VERBOSE) std::cout << "First layer weights / biases created" << std::endl << std::flush;
    // - perform the 2D convolution
    auto conv1 = tensorflow::ops::Conv2D(root,
                                  x,
                                  w1,
                                  {1, 1, 1, 1},
                                  "SAME");
    
    // - add the convolution and bias and then perform ReLU activation
    auto relu1 = tensorflow::ops::Relu(root,
                                tensorflow::ops::Add(root,
                                                     conv1,
                                                      b1));
    
    // - apply the max_pool layer
    auto h1 = tensorflow::ops::MaxPool(root,
                                relu1,
                                {1, 2, 2, 1},
                                {1, 2, 2, 1},
                                "SAME");
    
    if(VERBOSE) std::cout << "Creating the second layer" << std::endl << std::flush;
    // create the second convolutional layer
    // - create the weight variable
    auto w2 = weight_variable(root,
                                                   session,
                                                   {5, 5, 32, 64});
    // - create the bias variable
    auto b2 = bias_variable(root,
                                                 session,
                                                 {64});
    // - perform the 2D convolution
    auto conv2 = tensorflow::ops::Conv2D(root,
                                  h1,
                                  w2,
                                  {1, 1, 1, 1},
                                  "SAME");
    // - add the convolution and bias and then perform ReLU activation
    auto relu2 = tensorflow::ops::Relu(root,
                                tensorflow::ops::Add(root,
                                                     conv2,
                                                     b2));
    // - apply the max_pool layer
    auto h2 = tensorflow::ops::MaxPool(root,
                                relu2,
                                {1, 2, 2, 1},
                                {1, 2, 2, 1},
                                "SAME");

    if(VERBOSE) std::cout << "Creating the first fc layer" << std::endl << std::flush;
    // create the fully connected layer
    // - create the weight variable
    auto w_fc1 = weight_variable(root,
                                                      session,
                                                      {7 * 7 * 64, 1024});
    // - create the bias variable
    auto b_fc1 = bias_variable(root,
                                                    session,
                                                    {1024});

    // - create a flattened h2
    auto h2_flat = tensorflow::ops::Reshape(root,
                                     h2,
                                     {-1, 7 * 7 * 64});

    // - do the matrix multiplication
    auto h_fc1 = tensorflow::ops::Relu(root,
                                tensorflow::ops::Add(root,
                                                     tensorflow::ops::MatMul(root,
                                                                             h2_flat,
                                                                             w_fc1),
                                                     b_fc1));

    // FIXME add dropout here
    if(VERBOSE) std::cout << "Creating the output layer" << std::endl << std::flush;

    // create the output latyer
    // - create the weight variable
    auto w_fc2 = weight_variable(root,
                                                      session,
                                                      {1024, 10});

    // - create the bias variable
    auto b_fc2 = bias_variable(root,
                                                    session,
                                                    {10});

    // - do the matrix multiplication
    auto y_conv = tensorflow::ops::Add(root,
                                tensorflow::ops::MatMul(root,
                                                        h_fc1,
                                                        w_fc2),
                                b_fc2);
    // return the y_conv
    return y_conv;
}

int main(int argc,
         char **argv)
{
    // define the Images
    Image trainImages("data/train-images-idx3-ubyte");
    Image testImages("data/t10k-images-idx3-ubyte");
    // read the images
    trainImages.read();
    testImages.read();

    // define the labels
    Label trainLabels("data/train-labels-idx1-ubyte");
    Label testLabels("data/t10k-labels-idx1-ubyte");
    // read the labels
    trainLabels.read();
    testLabels.read();

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root);
 
    // create a placeholder for X
    tensorflow::ops::Placeholder x(root.WithOpName("X"), 
                                   tensorflow::DT_DOUBLE, 
                                   tensorflow::ops::Placeholder::Attrs().Shape(tensorflow::PartialTensorShape({-1, 28, 28, 1})));

    // create a placeholder for y
    tensorflow::ops::Placeholder y(root.WithOpName("y"),
                                   tensorflow::DT_UINT8,
                                   tensorflow::ops::Placeholder::Attrs().Shape(tensorflow::PartialTensorShape({-1})));

    tensorflow::Output y_conv = deepnn(root,
                                       session,
                                       x);

    if(VERBOSE) std::cout << "Returned from deepnn" << std::endl << std::flush;

    // std::cout << y_conv.dim_size(0) << " " << y_conv.dim_size(1) << std::endl;
    /* 
    y_conv = tensorflow::ops::Print(root, 
                                    y_conv, 
                                    {y_conv}, 
                                    tensorflow::ops::Print::Attrs().Summarize(1000));
    */

    auto y_one_hot = tensorflow::ops::OneHot(root,
                                             y,
                                             10,
                                             1.0,
                                             0.0);

    auto y_one_hotp = tensorflow::ops::Print(root, y_one_hot, (const std::initializer_list<tensorflow::Input>){y_one_hot}, tensorflow::ops::Print::Attrs().Summarize(1000));
    // y_conv = tensorflow::ops::Print(root, y_conv, { y_conv }, tensorflow::ops::Print::Attrs().Summarize(1000));

    // create the cross entropy loss
    auto loss = tensorflow::ops::SoftmaxCrossEntropyWithLogits(root,
                                                               y_conv,
                                                               y_one_hotp);
    // create the adam optimizer
    std::cout << root.status().ToString() << std::endl;
    tensorflow::ops::ApplyAdam optimizer(root,
                                         loss.loss,   // var to be optimized
                                         1.0,      // m
                                         1.0,      // v
                                         0.9,                               // beta1_power - FIXME confirm choice
                                         0.999,                             // beta2_power - FIXME confirm choice
                                         0.0001,                            // learning rate - taken from the deep mnist tutorial
                                         0.9,                               // beta1 - taken from the Python implementation's default values
                                         0.999,                             // beta2 - taken from the Python implementation's default values
                                         1e-8,                              // epsilon - taken from the Python implementation's default values
                                         loss.backprop);                              // grad - FIXME confirm choice
    std::cout << root.status().ToString() << std::endl;


    std::vector<tensorflow::Tensor> outputs;
    tensorflow::ClientSession::FeedType feed;
    int i;
    for(i=0;i<2;++i){
        
        if(VERBOSE) std::cout << "Iteration " << i << std::endl;
        if(i % 100 == 0) std::cout << "Iteration " << i << std::endl;
        // load a batch
        loadBatch(trainImages, trainLabels);

        // add the x to the tensorflow::ClientSession::FeedType feed
        feed.emplace(x, tensorflow::Input::Initializer(*(trainImages.batchTensor)));
        // add the y to the tensorflow::ClientSession::FeedType feed
        feed.emplace(y, tensorflow::Input::Initializer(*(trainLabels.batchTensor)));

        // run the session
        TF_CHECK_OK(session.Run(feed, { optimizer }, &outputs));

        // clear the tensorflow::ClientSession::FeedType
        feed.clear();
    }

    // std::cout << outputs[0].dtype() << std::endl;
    // std::cout << outputs[1].dtype() << std::endl;
    // std::cout << outputs[2].dtype() << std::endl;

    // TODO remove this is only for checking
    // output a layer of the tensor to check it was read properly
    /*
    auto flat = trainImages.batchTensor->flat<float>();
    std::ofstream out;
    out.open("output.txt", std::ios::out);
    int j;

    int numRows = trainImages.getNumRows();
    int numColumns = trainImages.getNumColumns();

    for (i=0;i<numRows;++i) {
        for(j=0;j<numColumns;++j) {
            out << std::setfill('0') << std::setw(3) << flat(j + i * numColumns + (numRows * numColumns * 12)) << " ";
            // out << flat(j + i * numColumns + (numRows * numColumns * 0)) << " ";
        }
        out << std::endl << std::flush;
    }
    out.close();
    */
    // end of TODO

    return 0;
}
