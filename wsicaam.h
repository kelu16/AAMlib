#ifndef WSICAAM_H
#define WSICAAM_H

#include "aam.h"

class WSICAAM : public AAM
{
public:
    WSICAAM();

    void train();
    float fit();
};

#endif // WSICAAM_H
