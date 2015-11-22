/*
 * Implementation of the Weighted Simultaneous Inverse Compositional algorithm as described in:
 * R. Gross, I. Matthews, S. Baker: Active Appearance Models with Occlusion
 */

#ifndef WSICAAM_H
#define WSICAAM_H

#include "aam.h"

class WSICAAM : public AAM
{
public:
    WSICAAM();

    void train();
    float fit();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // WSICAAM_H
