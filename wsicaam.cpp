#include "wsicaam.h"

#define fl at<float>

WSICAAM::WSICAAM()
{
}

void WSICAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    //AAM::calcSteepestDescentImages();

    this->initialized = true;
}

float WSICAAM::fit() {
    AAM::calcWarpedImage();
    AAM::calcErrorImage();
    if(this->steps == 0) {
        this->lambda = this->A*this->errorImage.t();
        AAM::calcErrorImage();
        AAM::calcSteepestDescentImages();
    }
    //AAM::calcSteepestDescentImages();

    Mat SD_sim;
    vconcat(this->steepestDescentImages, this->A, SD_sim);

    Mat weights = AAM::calcWeights();
    Mat weights2 = repeat(weights, SD_sim.rows, 1);

    Mat Hessian_sim = weights2.mul(SD_sim)*SD_sim.t();

    Mat deltaq = -Hessian_sim.inv()*weights2.mul(SD_sim)*this->errorImage.t();

    int numP = this->s.rows+this->s_star.rows;
    int numLambda = this->lambda.rows;
    Mat deltap = deltaq(cv::Rect(0,0,1,numP));
    Mat deltaLambda = deltaq(cv::Rect(0,numP,1,numLambda));

    AAM::updateAppearanceParameters(deltaLambda);
    AAM::updateInverseWarp(deltap);

    this->steps++;
    cout<<"Step: "<<this->steps<<endl;
    return sum(abs(deltaq))[0]/deltaq.rows;
}
