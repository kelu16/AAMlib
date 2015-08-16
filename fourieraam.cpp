#include "fourieraam.h"

FourierAAM::FourierAAM()
{
    this->type = "FourierAAM";
}

void FourierAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    //AAM::calcSteepestDescentImages();

    this->initialized = true;
}

float FourierAAM::fit() {
    AAM::calcWarpedImage();
    AAM::calcErrorImage();
    AAM::calcSteepestDescentImages();

    this->errorImage = this->calcFourier(this->errorImage);

    Mat SD_sim;
    vconcat(this->steepestDescentImages, this->A, SD_sim);

    for(int i=0; i<SD_sim.rows; i++) {
        Mat sd = this->calcFourier(SD_sim.row(i));
        sd.copyTo(SD_sim.row(i));
    }

    Mat Hessian_sim = SD_sim*SD_sim.t();

    Mat deltaq = -Hessian_sim.inv()*SD_sim*this->errorImage.t();

    int numP = this->s.rows+this->s_star.rows;
    int numLambda = this->lambda.rows;
    Mat deltap = deltaq(cv::Rect(0,0,1,numP));
    Mat deltaLambda = deltaq(cv::Rect(0,numP,1,numLambda));

    AAM::updateAppearanceParameters(deltaLambda);
    AAM::updateInverseWarp(deltap);

    return sum(abs(deltaq))[0]/deltaq.rows;
}

Mat FourierAAM::calcFourier(Mat img) {

    Mat out;

    dft(img, out);

    return out;
}

void FourierAAM::saveDataToFile(string fileName) {
    FileStorage fs(fileName, FileStorage::WRITE);

    AAM::saveDataToFileStorage(fs);

    fs.release();
}

void FourierAAM::loadDataFromFile(string fileName) {
    FileStorage fs(fileName, FileStorage::READ);

    if(!AAM::loadDataFromFileStorage(fs)) {
        return;
    }

    fs.release();

    this->initialized = true;
}
