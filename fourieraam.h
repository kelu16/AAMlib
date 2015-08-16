#ifndef FOURIERAAM_H
#define FOURIERAAM_H

#include "aam.h"

class FourierAAM : public AAM
{
public:
    FourierAAM();

    Mat calcFourier(Mat img);

    void train();
    float fit();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);
};

#endif // FOURIERAAM_H
