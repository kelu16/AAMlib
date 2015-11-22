/*
 * Implementation of the Simultaneous Inverse Compositional algorithm as described in:
 * R. Gross, I. Matthews, S. Baker:  Generic vs. Person Specific Active Appearance Models
 */

#ifndef SICAAM_H
#define SICAAM_H

#include "aam.h"

class SICAAM : public AAM
{
private:

public:
    SICAAM();

    void train();
    float fit();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // SICAAM_H
