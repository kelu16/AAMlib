#include "aam.h"

#define fl at<float>

AAM::AAM()
{
}

/*
 * Adds a set of training data consitisting of an image and a annotated shape vector
 */

void AAM::addTrainingData(const Mat &shape, const Mat &image) {
    Mat s = shape.reshape(1,1).t();
    Mat i = image.clone();

    switch(i.type()) {
    case CV_8UC3:
        i.convertTo(i, CV_32FC3);
        cvtColor(i,i,CV_BGR2GRAY);
        break;
    }

    //Mat i = image.reshape(1,1).t();

    if(this->trainingShapes.cols == 0) {
        this->trainingShapes = s;
        this->trainingImages.push_back(i);
    } else {
        hconcat(this->trainingShapes, s, this->trainingShapes);
        this->trainingImages.push_back(i);
    }
}

/*
 * Prepares the training shapes for PCA and calculates the shape vectors
 */

void AAM::calcShapeData() {
    cout<<"Started calc shape Data"<<flush;
    Mat pcaShapeData = this->trainingShapes.clone();
    Mat mean = AAM::alignShapeData(pcaShapeData);

    PCA shapePCA = PCA(pcaShapeData,
                       Mat(),
                       CV_PCA_DATA_AS_COL
                       );

    pcaShapeData.release();

    this->s0 = mean;
    this->numPoints = s0.rows/2;
    this->s = shapePCA.eigenvectors;

    if(!(this->numShapeParameters > 0  && this->numShapeParameters <= this->s.rows)) {
        float sumVariance = sum(shapePCA.eigenvalues)[0];
        float variance = 0.0f;
        for(int i=0; i<shapePCA.eigenvalues.rows; i++) {
            variance += shapePCA.eigenvalues.fl(i);
            if(variance/sumVariance >= this->targetShapeVariance) {
                cout<<"Num Shape Parameters: "<<i<<endl;
                this->numShapeParameters = i;
                break;
            }
        }
    }

    if(this->numShapeParameters > 0 && this->numShapeParameters <= this->s.rows) {
        this->s = this->s(cv::Rect(0,0,this->s.cols, this->numShapeParameters));
    }

    this->s_star = Mat::zeros(4, 2*this->numPoints, CV_32FC1);

    for(int i=0; i<this->numPoints; i++) {
        s_star.fl(0, 2*i) = s0.fl(2*i);
        s_star.fl(0, 2*i+1) = s0.fl(2*i+1);
        s_star.fl(1, 2*i) = -s0.fl(2*i+1);
        s_star.fl(1, 2*i+1) = s0.fl(2*i);
        s_star.fl(2, 2*i) = 1;
        s_star.fl(2, 2*i+1) = 0;
        s_star.fl(3, 2*i) = 0;
        s_star.fl(3, 2*i+1) = 1;
    }

    this->calcTriangleStructure(this->s0);
    this->calcTriangleMask();
    this->p = Mat::zeros(this->s.rows+this->s_star.rows, 1, CV_32FC1);
    cout<<endl;
}

/*
 * Prepares the training images for PCA and calculates the appearance images
 */

void AAM::calcAppearanceData() {
    cout<<"Started calc app Data"<<flush;
    int inputSize = this->trainingShapes.cols;
    Mat pcaAppearanceData = Mat(this->modelWidth*this->modelHeight,inputSize,trainingImages.at(0).type());

    for(int i=0; i<inputSize; i++) {
        Mat image = trainingImages.at(i).clone();
        normalize(image, image, 0, 1, NORM_MINMAX, image.type());

        if(this->preprocessImages == true) {
            switch(this->preprocessingMethod) {
            case AAM_PREPROC_TANTRIGGS:
                image = AAM::tanTriggsPreprocessing(image);
                break;
            case AAM_PREPROC_RETINEX:
                image = AAM::multiscaleRetinex(image);
                break;
            case AAM_PREPROC_DISTANCEMAPS:
                image = AAM::distanceMaps(image);
                break;
            }
        }

        image = AAM::warpImageToModel(image, trainingShapes.col(i));
        normalize(image, image, 0, 1, NORM_MINMAX, image.type(), this->triangleMask);

        image = image.reshape(1,1);

        if(i == 0) {
            pcaAppearanceData = image;
        } else {
            vconcat(pcaAppearanceData, image, pcaAppearanceData);
        }
        image.release();
    }

    Scalar mean;
    Scalar std;

    Mat tempMask = repeat(this->triangleMask.reshape(1,1), pcaAppearanceData.rows, 1);

    meanStdDev(pcaAppearanceData, mean, std, tempMask);

    this->standardDeviation = std[0];

    PCA appearancePCA = PCA(pcaAppearanceData,
                        Mat(),
                        CV_PCA_DATA_AS_ROW
                        );

    this->A0 = appearancePCA.mean;
    this->A = appearancePCA.eigenvectors;

    if(!(this->numAppParameters > 0  && this->numAppParameters <= this->s.rows)) {
        float sumVariance = sum(appearancePCA.eigenvalues)[0];
        float variance = 0.0f;
        for(int i=0; i<appearancePCA.eigenvalues.rows; i++) {
            variance += appearancePCA.eigenvalues.fl(i);
            if(variance/sumVariance >= this->targetAppVariance) {
                cout<<"Num Appearance Parameters: "<<i<<endl;
                this->numAppParameters = i;
                break;
            }
        }
    }

    if(this->numAppParameters > 0 && this->numAppParameters <= this->A.rows) {
        this->A = this->A(cv::Rect(0,0,this->A.cols, this->numAppParameters));
    }

    this->lambda = Mat::zeros(this->A.rows, 1, CV_32FC1);
    cout<<endl;
}

/*
 * Calculates the gradients of A0 and A
 */

void AAM::calcGradients() {
    this->calcGradX();
    this->calcGradY();
}

/*
 * Calculates the gradients of A0 and A in direction of X
 */

void AAM::calcGradX() {
    gradX = Mat::zeros(modelHeight, modelWidth, this->A0.type());

    Sobel( this->A0.reshape(1,this->modelHeight), gradX, this->A0.type(), 1, 0, 1);
    gradX = gradX.reshape(1,1);

    gradXA = Mat::zeros(this->A.rows, this->A.cols, this->A.type());
    for(int i=0; i<this->A.rows; i++) {
        Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, this->A.type(), 1, 0, 1);

        gradXA.row(i) = grad.reshape(1,1);
    }
}

/*
 * Calculates the gradients of A0 and A in direction of Y
 */

void AAM::calcGradY() {
    gradY = Mat::zeros(modelHeight, modelWidth, this->A0.type());

    Sobel( this->A0.reshape(1, this->modelHeight), gradY, this->A0.type(), 0, 1, 1);
    gradY = gradY.reshape(1,1);

    gradYA = Mat::zeros(this->A.rows, this->A.cols, this->A.type());
    for(int i=0; i<this->A.rows; i++) {
        Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, this->A.type(), 0, 1, 1);

        gradYA.row(i) = grad.reshape(1,1);
    }
}

/*
 * Calculates the steepest descent images gradient*jacobian
 */

void AAM::calcSteepestDescentImages() {
    Mat sdImg;

    Mat X = gradX + (this->gradXA.t()*this->lambda).t();
    Mat Y = gradY + (this->gradYA.t()*this->lambda).t();

    for(int i=0; i<this->jacobians.rows/2; i++) {
        Mat descentImage = X.mul(this->jacobians.row(2*i)) + Y.mul(this->jacobians.row(2*i+1));
        sdImg.push_back(descentImage);
    }
    this->steepestDescentImages = sdImg;
}

/*
 * Calculates the Jacobian of the warp
 */

void AAM::calcJacobian() {
    Mat j = Mat::zeros(2*s.rows+2*s_star.rows, this->A.cols, this->A.type());

    for(int i=0; i<numPoints; i++) {
        Mat derivate = this->derivateWarpToPoint(i); //dw/dx
        derivate = derivate.reshape(1,1);

        for(int globalTrans=0; globalTrans<this->s_star.rows; globalTrans++) {
            j.row(2*globalTrans) += derivate*this->s_star.fl(globalTrans, 2*i);
            j.row(2*globalTrans+1) += derivate*this->s_star.fl(globalTrans,  2*i+1);
        }

        for(int shapeVector=0; shapeVector < this->s.rows; shapeVector++) {
            j.row(2*shapeVector+2*this->s_star.rows) += derivate*this->s.fl(shapeVector, 2*i); //x-component
            j.row(2*shapeVector+2*this->s_star.rows+1) += derivate*this->s.fl(shapeVector, 2*i+1); //y-component
        }
    }

    this->jacobians = j;
}

/*
 * Calculates the derivate of the warp to each point
 */

Mat AAM::derivateWarpToPoint(int vertexId) {
    Mat derivate = Mat::zeros(modelHeight, modelWidth, CV_32F);
    vector<int> tris = this->triangleLookup[vertexId];
    int numTriangles = tris.size();
    for(int j=0; j<numTriangles; j++) {
        int a,b,c;
        int triId = tris[j];
        a = this->triangles.at<int>(triId,0);
        b = this->triangles.at<int>(triId,1);
        c = this->triangles.at<int>(triId,2);

        this->setFirstPoint(vertexId, a, b, c);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Get the boundaries of the triangle
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                Point px;
                px.x = col;
                px.y = row;

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    //Only when point is inside triangle
                    float val = 1-alpha-beta;

                    if(val > 0) {
                        derivate.fl(row,col) = val;
                    }
                }
            }
        }
    }

    return derivate;
}

/*
 * Calculates the mean of a matrix m
 */

Point2f AAM::calcMean(const Mat &m) {
    int n = m.rows;
    Point2f mean = Point2f(0,0);

    for(int j = 0; j < n; j++) {
        mean.x += m.fl(j,0);
        mean.y += m.fl(j,1);
    }

    mean.x /= n;
    mean.y /= n;

    return mean;
}

/*
 * Moves the center of mass of a matrix m to the origin
 */

Mat AAM::moveToOrigin(const Mat &m) {
    Mat Out = m.clone();
    int n = Out.rows;

    Point2f mean = this->calcMean(Out);

    for(int j = 0; j < n; j++) {
        Out.fl(j,0) = Out.fl(j,0) - mean.x;
        Out.fl(j,1) = Out.fl(j,1) - mean.y;
    }

    return Out;
}

/*
 * Aligns two shapes by performing procrustes analysis
 */

Mat AAM::procrustes(const Mat &X, const Mat &Y) {
    int n = X.rows/2;

    //Reshape to have col(0)=x and col(1)=y
    Mat X0 = X.reshape(1,n);
    Mat Y0 = Y.reshape(1,n);

    //move center to (0,0)
    X0 = this->moveToOrigin(X0);
    Y0 = this->moveToOrigin(Y0);

    float normX = sqrt(sum(X0.mul(X0))[0]);
    float normY = sqrt(sum(Y0.mul(Y0))[0]);

    X0 /= normX;
    Y0 /= normY;

    Mat U,Vt,D;
    Mat M = X0.t()*Y0;
    SVD::compute(M,D,U,Vt);

    //Rotation
    Mat R = Vt.t()*U.t();

    Mat x = Y0.col(0).clone();
    Mat y = Y0.col(1).clone();

    Y0.col(0) = R.fl(0,0)*x + R.fl(1,0)*y;
    Y0.col(1) = R.fl(0,1)*x + R.fl(1,1)*y;

    //Scaling
    float scaling = sum(D)[0];

    Mat Out = normX*scaling*Y0;


    Point2f meanX = calcMean(X.reshape(1,n));

    for(int i=0; i<n; i++) {
        Out.fl(i,0) += meanX.x;
        Out.fl(i,1) += meanX.y;
    }

    Out = Out.reshape(1,2*n);

    return Out;
}

/*
 * Aligns all training shapes using generalized procrustes analysis
 * Returns the mean shape
 */
Mat AAM::alignShapeData(Mat &shapeData) {
    Mat S = shapeData.clone();
    int numPoints = S.rows;
    int numShapes = S.cols;

    Mat meanShape = S.col(0).clone();
    Mat referenceShape = S.col(0).clone();

    float meanChange = 20.0f;

    while(meanChange > 0.1) {
        for(int i=0; i<numShapes; i++) {
            Mat Y = procrustes(meanShape.clone(), S.col(i).clone());
            Y.copyTo(S.col(i));
        }

        Mat newMeanShape = Mat::zeros(numPoints,1,CV_32FC1);

        for(int i=0; i<numPoints; i++) {
            float meanVal = 0;
            for(int j=0; j<numShapes; j++) {
                meanVal += S.fl(i,j);
            }
            meanVal /= numShapes;
            newMeanShape.fl(0,i) = meanVal;
        }

        Mat Y = procrustes(referenceShape, newMeanShape);
        newMeanShape = Y;

        meanChange = sum(abs(meanShape-newMeanShape))[0];

        meanShape = newMeanShape;
        newMeanShape.release();
    }

    Mat tempShape = meanShape.clone().reshape(1,numPoints/2);

    double minX,maxX,minY,maxY;

    minMaxIdx(tempShape.col(0), &minX, &maxX);
    minMaxIdx(tempShape.col(1), &minY, &maxY);

    tempShape = tempShape - repeat((Mat_<float>(1,2)<<minX - 2, minY - 2),numPoints/2,1);

    this->modelWidth = ceil(maxX-minX)+11;
    this->modelHeight = ceil(maxY-minY)+10;

    meanShape = tempShape.reshape(1, numPoints);

    S = S - repeat(meanShape,1,numShapes);

    shapeData = S;
    return meanShape;
}

/*
 * Calculates the traingle structure of a shape s using Delaunay triangulation
 */

void AAM::calcTriangleStructure(const Mat &s) {
    vector<Vec6f> triangleList;
    bool insert;
    Subdiv2D subdiv;
    int counter = 0;

    subdiv.initDelaunay(Rect(0,0,this->modelWidth, this->modelHeight));

    for(int i=0; i<this->numPoints; i++) {
        Point2f v = this->getPointFromMat(s, i);
        subdiv.insert(v);
    }

    subdiv.getTriangleList(triangleList);

    triangleLookup.clear();

    for(int i=0; i<this->numPoints; i++) {
        vector <int> temp;
        this->triangleLookup.push_back(temp);
    }

    this->triangles = Mat(triangleList.size(), 3, CV_32S);

    for(unsigned int i=0; i<triangleList.size(); i++) {
        Vec6f t = triangleList[i];
        vector<Point2f> pt(3);

        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5]);

        insert = true;

        for(int j=0; j<3; j++) {
            if(pt[j].x > modelWidth || pt[j].y > modelHeight || pt[j].x < 0 || pt[j].y < 0) {
                insert = false;
                break;
            }
        }

        if(insert) {
            if(pt[0]!=pt[1] && pt[0]!=pt[2] && pt[1]!=pt[2]) {
                int posA, posB, posC;
                posA = this->findPointInShape(pt[0]);
                posB = this->findPointInShape(pt[1]);
                posC = this->findPointInShape(pt[2]);

                this->triangles.at<int>(counter, 0) = posA;
                this->triangles.at<int>(counter, 1) = posB;
                this->triangles.at<int>(counter, 2) = posC;

                this->triangleLookup[posA].push_back(counter);
                this->triangleLookup[posB].push_back(counter);
                this->triangleLookup[posC].push_back(counter);

                counter++;
            }
        }
    }

    this->triangles = this->triangles(cv::Rect(0,0,3,counter));
}

/*
 * Creates a lookup vector which contains the triangles for every vertex in the shape
 */

void AAM::calcTriangleLookup() {
    this->triangleLookup.clear();
    for(int i=0; i<this->numPoints; i++) {
        vector <int> temp;
        this->triangleLookup.push_back(temp);
    }

    for(int i=0; i<this->triangles.rows; i++) {
        int posA, posB, posC;
        posA = this->triangles.at<int>(i, 0);
        posB = this->triangles.at<int>(i, 1);
        posC = this->triangles.at<int>(i, 2);

        this->triangleLookup[posA].push_back(i);
        this->triangleLookup[posB].push_back(i);
        this->triangleLookup[posC].push_back(i);
    }
}

/*
 * Calculates the alpha and beta values used in the warp for every pixel inside the image
 * Additionally a mapping of each pixel to the triangle it is inside is computed
 * IMPORTANT: this mapping starts with a triangle index of 1 because a value of 0 is used for pixels not inside a triangle
 */

void AAM::calcTriangleMask() {
    Mat mask = Mat::zeros(this->modelHeight, this->modelWidth, CV_8UC1);
    Mat aMap = Mat::zeros(this->modelHeight, this->modelWidth, CV_32FC1);
    Mat bMap = Mat::zeros(this->modelHeight, this->modelWidth, CV_32FC1);
    Mat tMasks = Mat::zeros(this->triangles.rows, this->modelHeight*this->modelWidth, CV_8UC1);
    //Mat tmask = Mat::zeros(this->modelHeight, this->modelWidth, CV_32SC1);

    for(int i=0; i<this->triangles.rows; i++) {
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = this->getPointFromMat(this->s0, a);
        pb = this->getPointFromMat(this->s0, b);
        pc = this->getPointFromMat(this->s0, c);

        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                Point2f px(col, row);

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;
                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    aMap.at<float>(row, col) = alpha;
                    bMap.at<float>(row, col) = beta;
                    mask.at<unsigned char>(row,col) = i+1;
                    //tmask.at<int>(row,col) = i+1;
                    tMasks.at<unsigned char>(i, row*this->modelWidth+col) = 255;
                }
            }
        }
    }

    this->triangleMask = mask;
    this->triangleMasks = tMasks;
    this->alphaMap = aMap;
    this->betaMap = bMap;
}

/*
 * Checks if a point px is inside a triangle specified by three points pa,pb,pc
 */

bool AAM::isPointInTriangle(Point2f px, Point2f pa, Point2f pb, Point2f pc) {
    float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

    float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
    float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

    return ((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1));
}

/*
 * Returns the id of a point p inside the base shape s0
 */

int AAM::findPointInShape(const Point2f &p) {
    for(int i=0; i<this->numPoints; i++) {
        Point2f s = getPointFromMat(this->s0, i);
        if(s == p) {
            return i;
        }
    }

    return -1;
}

/*
 * Returns the point with the id pointId from the matrix m
 */

Point2f AAM::getPointFromMat(const Mat &m, int pointId) {
    return Point2f(m.fl(2*pointId),m.fl(2*pointId+1));
}

/*
 * Warps an image inputImage to the model space based on a shape inputPoints
 */

Mat AAM::warpImageToModel(const Mat &inputImage, const Mat &inputPoints) {
    Mat out = Mat::zeros(modelHeight, modelWidth, inputImage.type());

    for(int row=0; row<out.rows; row++) {
        for(int col=0; col<out.cols; col++) {
            int triId = this->triangleMask.at<unsigned char>(row, col)-1;
            if(triId >= 0) {
                Point2f srcTri[3];
                int a,b,c;
                a = this->triangles.at<int>(triId,0);
                b = this->triangles.at<int>(triId,1);
                c = this->triangles.at<int>(triId,2);

                srcTri[0] = this->getPointFromMat(inputPoints, a);
                srcTri[1] = this->getPointFromMat(inputPoints, b);
                srcTri[2] = this->getPointFromMat(inputPoints, c);

                Point px = srcTri[0] + this->alphaMap.fl(row, col)*(srcTri[1]-srcTri[0]) + this->betaMap.fl(row, col)*(srcTri[2]-srcTri[0]);

                if((px.x >= 0 && px.x < inputImage.cols) && (px.y >= 0 && px.y < inputImage.rows)) {
                    out.fl(row, col) = inputImage.fl(px.y, px.x);
                }
            }
        }
    }

    return out;
}

/*
 * Warps an image inputImage from the shape inputPoints to the shape outputPoints on the image outputImage
 */

Mat AAM::warpImage(const Mat &inputImage, const Mat &inputPoints, const Mat &outputImage, const Mat &outputPoints) {
    Mat warpedImage = outputImage;

    int triSize = this->triangles.rows;
    for(int i=0; i<triSize; i++) {
        Point2f srcTri[3], dstTri[3];
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        srcTri[0] = this->getPointFromMat(inputPoints, a);
        srcTri[1] = this->getPointFromMat(inputPoints, b);
        srcTri[2] = this->getPointFromMat(inputPoints, c);

        dstTri[0] = this->getPointFromMat(outputPoints, a);
        dstTri[1] = this->getPointFromMat(outputPoints, b);
        dstTri[2] = this->getPointFromMat(outputPoints, c);

        int min_x = floor(min(dstTri[0].x, min(dstTri[1].x, dstTri[2].x)));
        int max_x = ceil(max(dstTri[0].x, max(dstTri[1].x, dstTri[2].x)));
        int min_y = floor(min(dstTri[0].y, min(dstTri[1].y, dstTri[2].y)));
        int max_y = ceil(max(dstTri[0].y, max(dstTri[1].y, dstTri[2].y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                Point px(col, row);
                float den = (dstTri[1].x - dstTri[0].x)*(dstTri[2].y - dstTri[0].y)-(dstTri[1].y - dstTri[0].y)*(dstTri[2].x - dstTri[0].x);

                float alpha = ((px.x - dstTri[0].x)*(dstTri[2].y - dstTri[0].y)-(px.y - dstTri[0].y)*(dstTri[2].x - dstTri[0].x))/den;
                float beta = ((px.y - dstTri[0].y)*(dstTri[1].x - dstTri[0].x)-(px.x - dstTri[0].x)*(dstTri[1].y - dstTri[0].y))/den;

                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    Point pxa = srcTri[0] + alpha*(srcTri[1]-srcTri[0]) + beta*(srcTri[2]-srcTri[0]);

                    if((pxa.x >= 0 && pxa.x < inputImage.cols) && (pxa.y >= 0 && pxa.y < inputImage.rows)) {
                        warpedImage.fl(row, col) = inputImage.fl(pxa.y, pxa.x);
                    }
                }
            }
        }
    }

    return warpedImage;
}

/*
 * Defines a static number of shape parameters
 * IMPORTANT: Has to be set before execution calcShapeParams()
 */

void AAM::setNumShapeParameters(int num) {
    this->numShapeParameters = num;
}

/*
 * Defines a static number of appearance parameters
 * IMPORTANT: Has to be set before execution calcAppParams()
 */

void AAM::setNumAppParameters(int num) {
    this->numAppParameters = num;
}

/*
 * Defines the shape variance of the training set the shape parameters should represent
 * IMPORTANT: Has to be set before execution calcShapeParams()
 */

void AAM::setTargetShapeVariance(float var) {
    this->targetShapeVariance = var;
}

/*
 * Defines the appearance variance of the training set the appearance parameters should represent
 * IMPORTANT: Has to be set before execution calcAppParams()
 */

void AAM::setTargetAppVariance(float var) {
    this->targetAppVariance = var;
}

/*
 * Sets the image the AAM is fit to to fittingImage
 */

void AAM::setFittingImage(const Mat &fittingImage) {
    fittingImage.convertTo(this->fittingImage, CV_32FC3);
    cvtColor(this->fittingImage, this->fittingImage, CV_BGR2GRAY);

    normalize(this->fittingImage, this->fittingImage, 0, 1, NORM_MINMAX, this->fittingImage.type());
    if(this->preprocessImages == true) {
        this->preprocessedImage = this->fittingImage.clone();
        switch(this->preprocessingMethod) {
        case AAM_PREPROC_TANTRIGGS:
            this->preprocessedImage = AAM::tanTriggsPreprocessing(this->preprocessedImage);
            break;
        case AAM_PREPROC_RETINEX:
            this->preprocessedImage = AAM::multiscaleRetinex(this->preprocessedImage);
            break;
        case AAM_PREPROC_DISTANCEMAPS:
            this->preprocessedImage = AAM::distanceMaps(this->preprocessedImage);
            break;
        }
        normalize(this->preprocessedImage, this->preprocessedImage, 0, 1, NORM_MINMAX, this->preprocessedImage.type());
    }

    normalize(this->fittingImage, this->fittingImage, 0, 1, NORM_MINMAX, this->fittingImage.type());
}

/*
 * Sets the current shape of the AAM to shape
 */

void AAM::setStartingShape(const Mat &shape) {
    this->fittingShape = shape;
}

/*
 * Resets the estimation of the model parameters
 */

void AAM::resetParameters() {
    this->lambda = Mat::zeros(this->A.rows, 1, CV_32FC1);
}

/*
 * Resets the shape on the fitting image by using Viola-Jones face Detection
 * scalingParameter defines the initial scale of the shape inside the found face bounding box
 */

void AAM::resetShape(float scalingParameter) {
    this->resetParameters();

    CascadeClassifier faceCascade;
    vector<Rect> faces;
    Mat detectImage;

    this->fittingImage.convertTo(detectImage, CV_8UC1, 255);

    faceCascade.load("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml");
    faceCascade.detectMultiScale( detectImage, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );

    if(faces.size() > 0) {
        Mat fitPoints = this->s0.clone();

        Rect face = faces[0];

        fitPoints = fitPoints.reshape(1, this->numPoints);
        Mat x = fitPoints.col(0);
        Mat y = fitPoints.col(1);

        double minX, minY, maxX, maxY;
        minMaxIdx(x, &minX, &maxX);
        minMaxIdx(y, &minY, &maxY);

        double scale = face.width/(maxX-minX);
        scale *= scalingParameter; //For best initial fit, depends on the used model

        //Center at (0,0)
        x = x-minX-(maxX-minX)/2;
        y = y-minY-(maxY-minY)/2;

        //Scale
        x *= scale;
        y *= scale;

        //Move to Position of found Face;
        x += face.x+face.width/2;
        y += face.y+face.height/2;

        this->fittingShape = fitPoints.reshape(1, this->numPoints*2);
    } else {
        Rect face = Rect(this->fittingImage.cols/2-100, this->fittingImage.rows/2-100, 200, 200);

        Mat fitPoints = this->s0.clone();

        fitPoints = fitPoints.reshape(1, this->numPoints);
        Mat x = fitPoints.col(0);
        Mat y = fitPoints.col(1);

        double minX, minY, maxX, maxY;
        minMaxIdx(x, &minX, &maxX);
        minMaxIdx(y, &minY, &maxY);

        double scale = face.width/(maxX-minX);
        scale *= scalingParameter; //For best initial fit, depends on the used model

        //Center at (0,0)
        x = x-minX-(maxX-minX)/2;
        y = y-minY-(maxY-minY)/2;

        //Scale
        x *= scale;
        y *= scale;

        //Move to Position of found Face;
        x += face.x+face.width/2;
        y += face.y+face.height/2;

        this->fittingShape = fitPoints.reshape(1, this->numPoints*2);
    }

    this->steps = 0;
}

/*
 * Returns the average error per pixel between the AAM instance and the fitting image
 */

double AAM::getErrorPerPixel() {
    return sum(abs(this->errorImage))[0]/(countNonZero(this->errorImage));
}

/*
 * Returns the error image between the AAM instance and the fitting image
 */

Mat AAM::getErrorImage() {
    return this->errorImage.clone();
}

/*
 * Returns the shape of the AAM
 */

Mat AAM::getFittingShape() {
    return this->fittingShape.clone();
}

/*
 * Re-orders the points with the ids a,b,c so that a has the id id
 */

void AAM::setFirstPoint(int id, int &a, int &b, int &c) {
    if(a != id) {
        int temp = a;
        if(b == id) {
            a = b;
            b = temp;
        } else if(c == id) {
            a = c;
            c = temp;
        } else {
            cout<<"Error: Point not in Triangle "<<id<<" "<<a<<" "<<b<<" "<<c<<endl;
        }
    }
}

/*
 * Returns the combination of the base appearance with the weighted appearance images
 */

Mat AAM::getAppearanceReconstruction() {
    Mat appVar = this->A0 + this->lambda.t()*this->A;

    return appVar.reshape(1, this->modelHeight);
}

/*
 * Returns the combination of the base appearance with the weighted appearance images warped onto the fitting image
 */

Mat AAM::getAppearanceReconstructionOnFittingImage() {
    Mat out;
    Mat app = this->getAppearanceReconstruction();
    if(this->preprocessImages) {
        out = this->preprocessedImage.clone();
    } else {
        out = this->fittingImage.clone();
    }
    return this->warpImage(app, this->s0, out, this->fittingShape);
}

/*
 * Updates the apperance parameters with deltaLambda
 */

void AAM::updateAppearanceParameters(Mat deltaLambda) {
    this->lambda += deltaLambda;
}

/*
 * Updates the inverse warp with deltaShapeParam
 */

void AAM::updateInverseWarp(Mat deltaShapeParam) {
    Mat deltaq = deltaShapeParam(cv::Rect(0,0,1,4));
    Mat deltap = deltaShapeParam(cv::Rect(0,4,1,deltaShapeParam.rows-4));

    //Inverse Global Shape Transformation
    Mat A(2,2,CV_32FC1);
    Mat t(1,2,CV_32FC1);

    A.fl(0,0) = 1/(1 + deltaq.fl(0));
    A.fl(1,0) = deltaq.fl(1);
    A.fl(0,1) = -deltaq.fl(1);
    A.fl(1,1) = 1/(1 + deltaq.fl(0));

    t.fl(0,0) = -deltaq.fl(2);
    t.fl(0,1) = -deltaq.fl(3);

    float warpChange = sum(abs(deltaq))[0];

    //Lower Values improve the fitting stability, but increase the runtime
    if(warpChange > 0.3) {
        Mat deltaS0 = this->s0.clone().reshape(1,this->numPoints);

        Mat Rot = deltaS0*A;
        Mat tr = repeat(t, this->numPoints, 1);
        deltaS0 = Rot + tr - deltaS0;

        deltaS0 = deltaS0.reshape(1,2*this->numPoints);

        this->fittingShape += deltaS0;
    } else {
        Mat deltaS0 = this->s0 - (deltap.t()*this->s).t();
        deltaS0 = deltaS0.reshape(1, this->numPoints);

        Mat Rot = deltaS0*A;
        Mat tr  = repeat(t, this->numPoints, 1);
        deltaS0 = Rot + tr;

        deltaS0 = deltaS0.reshape(1,1).t();

        for(int i=0; i<this->numPoints; i++) {
            Point2f update = Point2f(0,0);

            Point2f px = AAM::getPointFromMat(deltaS0,i);
            vector<int> triangles = this->triangleLookup[i];

            int numTriangles = triangles.size();
            for(int j=0; j<numTriangles; j++) {
                int a,b,c;
                int triId = triangles[j];
                a = this->triangles.at<int>(triId,0);
                b = this->triangles.at<int>(triId,1);
                c = this->triangles.at<int>(triId,2);

                this->setFirstPoint(i, a, b, c);    // Sort points

                Point2f pa,pb,pc;
                pa = AAM::getPointFromMat(this->s0, a);
                pb = AAM::getPointFromMat(this->s0, b);
                pc = AAM::getPointFromMat(this->s0, c);

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                pa = AAM::getPointFromMat(this->fittingShape, a);
                pb = AAM::getPointFromMat(this->fittingShape, b);
                pc = AAM::getPointFromMat(this->fittingShape, c);

                update += alpha*(pb-pa) + beta*(pc-pa);
            }

            update.x /= numTriangles;
            update.y /= numTriangles;

            this->fittingShape.fl(2*i) += update.x;
            this->fittingShape.fl(2*i+1) += update.y;
        }
    }
}

/*
 * Warps the fitting image to the model space
 */

void AAM::calcWarpedImage() {
    if(this->preprocessImages == true) {
        this->warpedImage = AAM::warpImageToModel(this->preprocessedImage, this->fittingShape);
        switch(this->preprocessingMethod) {
        case AAM_PREPROC_HISTOGRAMMATCHING:
            Mat temp = AAM::histogramFitting(this->warpedImage, this->A0.reshape(1,this->modelHeight));
            this->warpedImage = Mat::zeros(temp.size(), temp.type());
            temp.copyTo(this->warpedImage, this->triangleMask);
            break;
        }
    } else {
        this->warpedImage = AAM::warpImageToModel(this->fittingImage, this->fittingShape);
    }

    normalize(this->warpedImage, this->warpedImage, 0, 1, NORM_MINMAX, this->warpedImage.type(), this->triangleMask);
}

/*
 * Calculates the error between the AAM and the fitting image
 */

void AAM::calcErrorImage() {
    Mat errorImage = AAM::getAppearanceReconstruction() - this->warpedImage;
    this->errorImage = errorImage.reshape(1,1);
}

/*
 * Calculates the weights of each pixel in the error image with a robust error function
 */

void AAM::calcErrorWeights() {
    this->errorWeights = this->calcWeights();
}

/*
 * Sets the used error function to function
 */

void AAM::setErrorFunction(int function) {
    this->errorFunction = function;
}

/*
 * Sets the used preprocessing method to method
 * IMPORTANT: has to be set before executing calcAppData()
 */

void AAM::setProcessingMethod(int method) {
    this->preprocessingMethod = method;
}

/*
 * Calculates the weights of each pixel in the error image with a robust error function
 */

Mat AAM::calcWeights() {
    Mat i = this->errorImage.clone();

    this->outliers = Mat::zeros(i.rows, i.cols, CV_32FC1);

    float stdPi = 1/(standardDeviation*sqrt(2*M_PI));
    float stdSq = 2*standardDeviation*standardDeviation;

    for(int row=0; row<i.rows; row++) {
        for(int col=0; col<i.cols; col++) {
            float val = i.fl(row, col);
            float absValue = abs(val);

            switch(this->errorFunction) {
                default:
                case AAM_ERR_CAUCHY:
                    i.fl(row, col) = 1/(1+pow(val/(0.1),2));
                    break;
                case AAM_ERR_GEMANMCCLURE:
                    i.fl(row, col) = 1/pow(1+pow(val,2),2);
                    break;
                case AAM_ERR_HUBER: {
                    float c = 0.05;
                    if(absValue > c) {
                        i.fl(row, col) = c/absValue;
                    } else {
                        i.fl(row, col) = 1;
                    }
                    break;
                }
                case AAM_ERR_WELSCH:
                    i.fl(row, col) = exp(-pow(val/(standardDeviation),2));
                    break;
                case AAM_ERR_TUKEY: {
                    float c = 0.2;
                    if(absValue < c) {
                        i.fl(row, col) =  pow(1-pow(val/(c),2),2);
                    } else {
                        i.fl(row, col) = 0;
                    }
                    break;
                }
                case AAM_ERR_EXPONENTIAL:
                    i.fl(row, col) = (stdPi)*exp(-abs(i.fl(row, col))/(stdSq));
                    break;
            }
        }
    }

    return i;
}

/*
 * Returns if all precomputational steps are completed
 */

bool AAM::isInitialized() {
    return this->initialized;
}

/*
 * checks if a fitting image is set
 */

bool AAM::hasFittingImage() {
    return !this->fittingImage.empty();
}

/*
 * Saves the current model to a file fs
 */

bool AAM::saveDataToFileStorage(FileStorage fs) {
    fs << "type" << this->type;

    fs << "s0" << this->s0;
    fs << "s" << this->s;
    fs << "s_star" << this->s_star;

    fs << "A0" << this->A0;
    fs << "A" << this->A;

    fs << "numPoints" << this->numPoints;
    fs << "modelHeight" << this->modelHeight;
    fs << "modelWidth" << this->modelWidth;

    fs << "triangles" << this->triangles;
    fs << "triangleMask" << this->triangleMask;
    fs << "alphaMap" << this->alphaMap;
    fs << "betaMap" << this->betaMap;

    fs << "standardDeviation" << this->standardDeviation;

    //fs << "gradX" << this->gradX;
    //fs << "gradY" << this->gradY;
    //fs << "gradXA" << this->gradXA;
    //fs << "gradYA" << this->gradYA;
    //fs << "jacobians" << this->jacobians;
    fs << "steepestDescentImages" << this->steepestDescentImages;

    return true;
}

/*
 * Loads the model from a file fs
 */

bool AAM::loadDataFromFileStorage(FileStorage fs) {
    if(fs["type"] != this->type) {
        cout<<"Wrong AAM type"<<endl;
        return false;
    }

    fs["s0"] >> this->s0;
    fs["s"] >> this->s;
    fs["s_star"] >> this->s_star;

    fs["A0"] >> this->A0;
    fs["A"] >> this->A;

    fs["numPoints"] >> this->numPoints;
    fs["modelHeight"] >> this->modelHeight;
    fs["modelWidth"] >> this->modelWidth;

    fs["triangles"] >> this->triangles;
    fs["triangleMask"] >> this->triangleMask;
    fs["alphaMap"] >> this->alphaMap;
    fs["betaMap"] >> this->betaMap;

    fs["standardDeviation"] >> this->standardDeviation;

    //fs["gradX"] >> this->gradX;
    //fs["gradY"] >> this->gradY;
    //fs["gradXA"] >> this->gradXA;
    //fs["gradYA"] >> this->gradYA;
    //fs["jacobians"] >> this->jacobians;
    fs["steepestDescentImages"] >> this->steepestDescentImages;

    this->lambda = Mat::zeros(this->A.rows, 1, CV_32FC1);
    this->calcTriangleLookup();

    return true;
}

/*
 * Returns the mean appearance reshaped to the original image size
 */

Mat AAM::getA0() {
    return this->A0.reshape(1,this->modelHeight);
}

/*
 * Defines if the images are preprocessed by the set method (see setProcessingMethod)
 * IMPORTANT: has to be set before the initialization of the model, because the training images need to be preprocessed as well
 */


void AAM::setPreprocessImages(bool on) {
    this->preprocessImages = on;
}

/*
 * Computes the Tan-Triggs-Representation of an image
 *
 * Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

Mat AAM::tanTriggsPreprocessing(InputArray src, float alpha, float tau, float gamma, int sigma0, int sigma1) {
    // Convert to floating point:
    Mat X = src.getMat();
    //X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);

    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        Mat exp_x, exp_negx;
    exp( I / tau, exp_x );
    exp( -I / tau, exp_negx );
    divide( exp_x - exp_negx, exp_x + exp_negx, I );
        I = tau * I;
    }

    return I;
}

/*
 * Performs a single scale retinex filter with a given kernel size
 */

Mat AAM::retinex(Mat image, int kernelSize) {
    Mat out, logImg;
    Mat kernel = getGaussianKernel(kernelSize, -1);
    kernel = kernel*kernel.t();

    kernel = kernel/sum(kernel)[0];

    normalize(image, image, 0, 1, NORM_MINMAX, CV_32FC1);

    image += 0.01;

    filter2D(image, out, image.depth(), kernel);

    log(image, logImg);
    log(abs(out), out);

    out = logImg-out;

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            out.fl(i,j) = max(0.0f, out.fl(i,j));
        }
    }

    return out;
}

/*
 * Performs a multiscaleRetinex filter with fixed values {15,80,250}
 */

Mat AAM::multiscaleRetinex(Mat image) {
    normalize(image, image, 0, 1, NORM_MINMAX, CV_32FC1);

    Mat ret = Mat::zeros(image.size(), image.type());

    int numFilters = 3;
    int kernelSizes[numFilters] = {15,80,250};

    for(int i=0; i<numFilters; i++) {
        ret += retinex(image, kernelSizes[i]);
    }

    ret = ret/numFilters;

    Mat logImg;
    log(image, logImg);
    Mat out = logImg+ret;

    return out;
}

/*
 * Matches the histogram of an input image I to a reference image R
 */

Mat AAM::histogramFitting(Mat I, Mat R) {
    Mat insrc,inref;
    I.convertTo(insrc, CV_8UC1, 255);
    normalize(insrc, insrc, 0, 255, NORM_MINMAX, CV_8UC1, this->triangleMask);
    R.convertTo(inref, CV_8UC1, 255);
    normalize(inref, inref, 0, 255, NORM_MINMAX, CV_8UC1, this->triangleMask);

    float range[] = {0,255};
    const float* histRange = {range};
    int histSize = 256;
    bool uniform = true;

    int numSubWindows = 2;

    int windowSize = I.cols/numSubWindows;

    Mat mappings[numSubWindows];

    for(int j=0; j<numSubWindows; j++) {
        Mat in, ref;
        Rect subWindow = Rect(j*windowSize,0,windowSize, insrc.rows);
        in = insrc(subWindow);
        ref = inref(subWindow);

        Mat inHist, refHist;
        calcHist(&in,1,0,this->triangleMask(subWindow),inHist,1,&histSize,&histRange, uniform, false);
        calcHist(&ref,1,0,this->triangleMask(subWindow),refHist,1,&histSize,&histRange, uniform, false);

        double min, max;
        minMaxLoc(refHist, &min, &max);
        refHist = refHist/max;
        minMaxLoc(inHist, &min, &max);
        inHist = inHist/max;

        Mat refAcc, inAcc;
        refHist.copyTo(refAcc);
        inHist.copyTo(inAcc);

        for(int i=1; i<histSize; i++) {
            refAcc.fl(i) += refAcc.fl(i-1);
            inAcc.fl(i) += inAcc.fl(i-1);
        }

        minMaxLoc(refAcc, &min, &max);
        refAcc = refAcc/max;
        minMaxLoc(inAcc, &min, &max);
        inAcc = inAcc/max;

        Mat Mv(1, 256, CV_8UC1);
        uchar last=0;

        for(int i=0; i<inAcc.rows; i++) {
            float F1 = inAcc.fl(i);

            for(uchar k=last; k<refAcc.rows; k++) {
                float F2 = refAcc.fl(k);
                if(abs(F2-F1) < 0.000001 || F2>F1) {
                    Mv.at<uchar>(i) = k;
                    last = k;
                    break;
                }
            }
        }

        Mat lut(1,256,CV_8UC1,Mv.ptr<uchar>());

        mappings[j] = lut.clone();
    }

    Mat left, right;

    LUT(insrc,mappings[0],left);
    LUT(insrc,mappings[1],right);

    for(int i=0; i<insrc.cols; i++) {
        float leftness = (insrc.cols-i)/(float)insrc.cols;
        insrc.col(i) = leftness*left.col(i) + (1-leftness)*right.col(i);
    }

    insrc.convertTo(insrc, CV_32FC1);

    return insrc;
}

/*
 * Transform an input image I to the distance of each pixel to the nearest edge
 */

Mat AAM::distanceMaps(Mat I) {
    Mat out = I.clone();
    out.convertTo(out, CV_8UC1,255);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setTilesGridSize(Size(4,4));
    clahe->apply(out,out);

    blur(out,out,Size(5,5));

    Mat edges = out.clone();

    Canny(edges,edges,30,120);

    edges = 1-edges;

    distanceTransform(edges,edges,CV_DIST_L2, 3);

    edges.convertTo(edges,CV_8UC1);
    normalize(edges, edges, 0, 255, NORM_MINMAX, CV_8UC1);

    edges.convertTo(out,CV_32FC1);

    return out;
}
