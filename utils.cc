#include <iostream>
#include <fstream>
#include <cstdio>

#include <arpa/inet.h>

#include "utils.h"

int readBigEndianInt(std::ifstream &in)
{
    Bint val;
    in.read(val.buffer, 4);
    val.num = htonl(val.num);
    
    std::cout << val.num << std::endl;
    printf("%02x\n", val.buffer[0]);
    printf("%02x\n", val.buffer[1]);
    printf("%02x\n", val.buffer[2]);
    printf("%02x\n", val.buffer[3]);

    return val.num;
}

