#include "robustaam.h"

#define fl at<float>

RobustAAM::RobustAAM()
{
    this->triangleShapeHessians.clear();
    this->triangleAppHessians.clear();
}

void RobustAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    AAM::calcSteepestDescentImages();

    this->calcTriangleHessians();

    this->initialized = true;
}

float RobustAAM::fit() {
    high_resolution_clock::time_point start_warpImage = high_resolution_clock::now();
    AAM::calcWarpedImage();
    high_resolution_clock::time_point end_warpImage = high_resolution_clock::now();
    cout<<"Warp Image: "<<duration_cast<microseconds>(end_warpImage-start_warpImage).count()<<endl;
    AAM::calcErrorImage();
    high_resolution_clock::time_point end_calcErrorImage = high_resolution_clock::now();
    cout<<"Calc Error Image: "<<duration_cast<microseconds>(end_calcErrorImage-end_warpImage).count()<<endl;
    AAM::calcErrorWeights();
    high_resolution_clock::time_point end_calcWeights = high_resolution_clock::now();
    cout<<"Calc Weights: "<<duration_cast<microseconds>(end_calcWeights-end_calcErrorImage).count()<<endl;

    Mat deltaLambda = this->calcAppeanceUpdate();
    high_resolution_clock::time_point end_calcAppUpdate = high_resolution_clock::now();
    cout<<"Calc App Update: "<<duration_cast<microseconds>(end_calcAppUpdate-end_calcWeights).count()<<endl;
    AAM::updateAppearanceParameters(deltaLambda);
    high_resolution_clock::time_point end_appUpdate = high_resolution_clock::now();
    cout<<"App Update: "<<duration_cast<microseconds>(end_appUpdate-end_calcAppUpdate).count()<<endl;

    AAM::calcErrorImage();
    high_resolution_clock::time_point end_calcErrorImage2 = high_resolution_clock::now();
    cout<<"Calc Error Image: "<<duration_cast<microseconds>(end_calcErrorImage2-end_appUpdate).count()<<endl;
    AAM::calcErrorWeights();
    high_resolution_clock::time_point end_calcWeights2 = high_resolution_clock::now();
    cout<<"Calc Weights: "<<duration_cast<microseconds>(end_calcWeights2-end_calcErrorImage2).count()<<endl;

    Mat deltaShapeParam = this->calcShapeUpdate();
    high_resolution_clock::time_point end_calcShapeParam = high_resolution_clock::now();
    cout<<"Calc Shape Param: "<<duration_cast<microseconds>(end_calcShapeParam-end_calcWeights2).count()<<endl;

    AAM::updateInverseWarp(deltaShapeParam);
    high_resolution_clock::time_point end_updateWarp = high_resolution_clock::now();
    cout<<"Update Warp: "<<duration_cast<microseconds>(end_updateWarp-end_calcShapeParam).count()<<endl;

    /*
    namedWindow("reconstruction");
    imshow("reconstruction", this->getAppearanceReconstructionOnFittingImage());
    */

    Mat parameterUpdates;
    vconcat(deltaLambda, deltaShapeParam, parameterUpdates);
    return sum(abs(parameterUpdates))[0]/parameterUpdates.rows;
}

void RobustAAM::calcTriangleHessians() {
    for(int i=0; i<this->triangles.rows; i++) {
        cout<<"calcTriangleHessian "<<i<<endl;
        Mat sdImg = this->steepestDescentImages.clone();
        Mat appSdImg = this->A.clone();


        /*
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        Mat sdImg = Mat::zeros(this->steepestDescentImages.size(), CV_32FC1);
        Mat appSdImg = Mat::zeros(this->A.size(), CV_32FC1);

        int n=0;

        for(int row=min_x; row<max_x; row++) {
            for(int col=min_y; col<max_y; col++) {
                //cout<<this->triangleMask.at<int>(row, col)<<endl;
                int triangleNumber = this->triangleMasks(Rect(row, col, 1, 1)).at<int>(0,0);
                //cout<<triangleNumber<<" "<<this->triangleMasks(Rect(row, col, 1, 1)).at<int>(0,0)<<endl;
                if(triangleNumber == (i+1)) {
                    this->steepestDescentImages.col(row*this->modelWidth+col).copyTo(sdImg.col(row*this->modelWidth+col));
                    this->A.col(row*this->modelWidth+col).copyTo(appSdImg.col(row*this->modelWidth+col));
                    n++;
                    //sdImg.col(row*this->modelWidth+col) = 0;
                    //appSdImg.col(row*this->modelWidth+col) = 0;
                } else {
                    //cout<<row<<" "<<col<<" "<<i+1<<" "<<this->triangleMasks.at<int>(row, col)<<endl;
                }
            }
        }
        cout<<"Number of Points: "<<n<<endl;
        if(n == 0) {
            //cout<<this->triangleMasks(cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y))<<endl;
            //cout<<"Triangle: "<<i+1<<" Points: "<<pa<<pb<<pc<<endl;
        }
        */

        //H^i_p
        for(int j=0; j<sdImg.rows; j++) {
            Mat sd;
            sdImg.row(j).copyTo(sd, this->triangleMasks.row(i));
            sd.copyTo(sdImg.row(j));
        }

        for(int j=0; j<appSdImg.rows; j++) {
            Mat sd;
            appSdImg.row(j).copyTo(sd, this->triangleMasks.row(i));
            sd.copyTo(appSdImg.row(j));
        }

        Mat Hessian = sdImg*sdImg.t();
        this->triangleShapeHessians.push_back(Hessian);

        //H^i_A

        Mat AppHessian = appSdImg*appSdImg.t();
        this->triangleAppHessians.push_back(AppHessian);
    }
}

Mat RobustAAM::calcWeightedHessian(vector<Mat> triangleHessians) {
    Mat Hessian = Mat::zeros(triangleHessians[0].rows, triangleHessians[0].cols, CV_32FC1);
    //Mat eImg = this->calcWeights();
    Mat eImg = this->errorWeights.reshape(1,this->modelHeight);

    for(int i=0; i<this->triangles.rows; i++) {
        float sum = 0.0f;
        float weight = 0.0f;
        int n = 0;
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                //if(this->triangleMask.at<unsigned char>(row, col) == i+1) {
                if(AAM::isPointInTriangle(Point(col, row), pa, pb, pc)) {
                    sum += eImg.at<float>(row,col);
                    n++;
                }
            }
        }

        if(n>0) {
            weight = (sum/(float)n);
        }
        //cout<<"Triangle "<<i<<" Weight "<<weight<<endl;
        Hessian += weight*triangleHessians[i];
    }

    return Hessian;
}

Mat RobustAAM::calcAppeanceUpdate() {
    Mat appHessian = this->calcWeightedHessian(this->triangleAppHessians);

    Mat weights = this->errorWeights.clone();
    weights = repeat(weights, this->A.rows, 1);

    Mat result = weights.mul(this->A);
    result = result*this->errorImage.t();

    Mat inv = -appHessian.inv();

    return inv*result;
    //return -appHessian.inv()*weights.mul(this->A)*this->errorImage.t();
}

Mat RobustAAM::calcShapeUpdate() {
    Mat shapeHessian = this->calcWeightedHessian(this->triangleShapeHessians);

    Mat weights = this->errorWeights.clone();
    weights = repeat(weights, this->steepestDescentImages.rows, 1);

    Mat result = weights.mul(this->steepestDescentImages);
    result = result*this->errorImage.t();

    Mat inv = -shapeHessian.inv();

    return inv*result;
}
