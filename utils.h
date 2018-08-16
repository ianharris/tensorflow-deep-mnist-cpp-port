#ifndef __MNIST_UTILS_H__
#define __MNIST_UTILS_H__

union Bint
{
    char buffer[4];
    uint32_t num;
};

// read a big endian int
int readBigEndianInt(std::ifstream &in);

#endif

