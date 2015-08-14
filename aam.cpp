#include "aam.h"

#define fl at<float>

AAM::AAM()
{
    this->numPoints = 0;
    this->numAppParameters = 20;
    this->numShapeParameters = 11;
    this->initialized = false;
    this->steps = 0;
}

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

void AAM::calcShapeData() {
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
}

void AAM::calcAppearanceData() {
    int inputSize = this->trainingShapes.cols;
    Mat pcaAppearanceData = Mat(this->modelWidth*this->modelHeight,inputSize,CV_32FC1);

    for(int i=0; i<inputSize; i++) {
        Mat image = trainingImages.at(i).clone();

        image = AAM::warpImageToModel(image, trainingShapes.col(i));
        normalize(image, image, 0, 1, NORM_MINMAX, CV_32FC1, this->triangleMask);

        waitKey();

        image = image.reshape(1,1).t();

        image.copyTo(pcaAppearanceData.col(i));
        image.release();
    }

    PCA appearancePCA = PCA(pcaAppearanceData,
                        Mat(),
                        CV_PCA_DATA_AS_COL
                        );

    this->A0 = appearancePCA.mean;
    //normalize(this->A0, this->A0, 0, 1, NORM_MINMAX, CV_32FC1);
    this->A = appearancePCA.eigenvectors;

    if(this->numAppParameters > 0 && this->numAppParameters <= this->A.rows) {
        this->A = this->A(cv::Rect(0,0,this->A.cols, this->numAppParameters));
    }

    this->lambda = Mat::zeros(this->A.rows, 1, CV_32FC1);
}

void AAM::calcGradients() {
    this->calcGradX();
    this->calcGradY();
}

void AAM::calcGradX() {
    gradX = Mat::zeros(modelHeight, modelWidth, CV_32FC1);

    Sobel( this->A0.reshape(1,this->modelHeight), gradX, CV_32FC1, 1, 0, 1);
    gradX = gradX.reshape(1,1);

    gradXA = Mat::zeros(this->A.rows, this->A.cols, CV_32FC1);
    for(int i=0; i<this->A.rows; i++) {
        Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, CV_32FC1, 1, 0, 1);

        gradXA.row(i) = grad.reshape(1,1);
    }
}

void AAM::calcGradY() {
    gradY = Mat::zeros(modelHeight, modelWidth, CV_32FC1);

    Sobel( this->A0.reshape(1, this->modelHeight), gradY, CV_32FC1, 0, 1, 1);
    gradY = gradY.reshape(1,1);

    gradYA = Mat::zeros(this->A.rows, this->A.cols, CV_32FC1);
    for(int i=0; i<this->A.rows; i++) {
        Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, CV_32FC1, 0, 1, 1);

        gradYA.row(i) = grad.reshape(1,1);
    }
}

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

void AAM::calcJacobian() {
    Mat j = Mat::zeros(2*s.rows+2*s_star.rows, this->A.cols, CV_32FC1);

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

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
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
                    //Nur wenn Punkt innerhalb des Dreiecks liegt
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

//returns mean shape
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

    this->modelWidth = ceil(maxX-minX)+4;
    this->modelHeight = ceil(maxY-minY)+4;

    meanShape = tempShape.reshape(1, numPoints);

    S = S - repeat(meanShape,1,numShapes);

    shapeData = S;
    return meanShape;
}

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
        pa = AAM::getPointFromMat(this->s0, a);
        pb = AAM::getPointFromMat(this->s0, b);
        pc = AAM::getPointFromMat(this->s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
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

bool AAM::isPointInTriangle(Point2f px, Point2f pa, Point2f pb, Point2f pc) {
    float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

    float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
    float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

    return ((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1));
}

int AAM::findPointInShape(const Point2f &p) {
    for(int i=0; i<this->numPoints; i++) {
        Point2f s = getPointFromMat(this->s0, i);
        if(s == p) {
            return i;
        }
    }

    return -1;
}

Point2f AAM::getPointFromMat(const Mat &m, int pointId) {
    return Point2f(m.fl(2*pointId),m.fl(2*pointId+1));
}

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

    //return warpImage(inputImage, inputPoints, out, this->s0);
}

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

        warpTextureFromTriangle(srcTri, inputImage, dstTri, warpedImage);
    }

    return warpedImage;
}

void AAM::warpTextureFromTriangle(Point2f srcTri[], const Mat &originalImage, Point2f dstTri[], Mat warp_final) {
    Mat warp_mat(2, 3, CV_32FC1);
    Mat warp_dst, warp_mask, srcImg;
    int smoothingParam = 1;

    int min_x_src = floor(min(srcTri[0].x, min(srcTri[1].x, srcTri[2].x)));
    int max_x_src = ceil(max(srcTri[0].x, max(srcTri[1].x, srcTri[2].x)));
    int min_y_src = floor(min(srcTri[0].y, min(srcTri[1].y, srcTri[2].y)));
    int max_y_src = ceil(max(srcTri[0].y, max(srcTri[1].y, srcTri[2].y)));

    int src_size_x = max(max_x_src - min_x_src,1)+2*smoothingParam;
    int src_size_y = max(max_y_src - min_y_src,1)+2*smoothingParam;

    srcImg = originalImage(cv::Rect_<int>(min_x_src-smoothingParam,min_y_src-smoothingParam,src_size_x,src_size_y));
    for(int i=0; i<3; i++) {
        srcTri[i] -= Point2f(min_x_src-smoothingParam, min_y_src-smoothingParam);
    }

    int min_x_dst = floor(min(dstTri[0].x, min(dstTri[1].x, dstTri[2].x)));
    int max_x_dst = ceil(max(dstTri[0].x, max(dstTri[1].x, dstTri[2].x)));
    int min_y_dst = floor(min(dstTri[0].y, min(dstTri[1].y, dstTri[2].y)));
    int max_y_dst = ceil(max(dstTri[0].y, max(dstTri[1].y, dstTri[2].y)));

    int dst_size_x = max_x_dst - min_x_dst;
    int dst_size_y = max_y_dst - min_y_dst;

    for(int i=0; i<3; i++) {
        dstTri[i] -= Point2f(min_x_dst, min_y_dst);
    }

    Point triPoints[3];
    triPoints[0] = dstTri[0];
    triPoints[1] = dstTri[1];
    triPoints[2] = dstTri[2];

    warp_dst = Mat::zeros(dst_size_y, dst_size_x, originalImage.type());
    warp_mask = Mat::zeros(dst_size_y, dst_size_x, CV_8U);

    // Get the Affine Transform
    warp_mat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform to the src image
    warpAffine(srcImg, warp_dst, warp_mat, warp_dst.size());
    fillConvexPoly(warp_mask, triPoints, 3, Scalar(255,255,255), CV_AA, 0);
    warp_dst.copyTo(warp_final(cv::Rect_<int>(min_x_dst, min_y_dst, dst_size_x, dst_size_y)), warp_mask);
}

void AAM::setNumShapeParameters(int num) {
    this->numShapeParameters = num;
}

void AAM::setNumAppParameters(int num) {
    this->numAppParameters = num;
}

void AAM::setFittingImage(const Mat &fittingImage) {
    fittingImage.convertTo(this->fittingImage, CV_32FC3);
    cvtColor(this->fittingImage, this->fittingImage, CV_BGR2GRAY);
    normalize(this->fittingImage, this->fittingImage, 0, 1, NORM_MINMAX, CV_32FC1);
}

void AAM::setStartingShape(const Mat &shape) {
    this->fittingShape = shape;
}

void AAM::resetShape() {
    this->lambda = Mat::zeros(this->A.rows, 1, CV_32FC1);

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
        scale *= 0.8; //For best initial fit, depends on the used model

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
        scale *= 0.8; //For best initial fit, depends on the used model

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

double AAM::getErrorPerPixel() {
    //return sum(abs(this->errorImage))[0]/(this->errorImage.rows*this->errorImage.cols);
    return sum(abs(this->errorImage))[0]/(countNonZero(this->errorImage));
}

Mat AAM::getErrorImage() {
    return this->errorImage;
}

Mat AAM::getFittingShape() {
    return this->fittingShape;
}

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

Mat AAM::getAppearanceReconstruction() {
    Mat appVar = this->A0 + this->A.t()*this->lambda;

    return appVar.reshape(1, this->modelHeight);
}

Mat AAM::getAppearanceReconstructionOnFittingImage() {
    Mat app = this->getAppearanceReconstruction();
    Mat out = this->fittingImage.clone();
    return this->warpImage(app, this->s0, out, this->fittingShape);
}

void AAM::updateAppearanceParameters(Mat deltaLambda) {
    this->lambda += deltaLambda;
}

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

    /*
    deltaq.fl(2) = deltaq.fl(2)/this->fittingImage.cols;
    deltaq.fl(3) = deltaq.fl(3)/this->fittingImage.rows;
    */
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

void AAM::calcWarpedImage() {
    this->warpedImage = AAM::warpImageToModel(this->fittingImage, this->fittingShape);
    normalize(this->warpedImage, this->warpedImage, 0, 1, NORM_MINMAX, CV_32FC1, this->triangleMask);
}

void AAM::calcErrorImage() {
    Mat errorImage = AAM::getAppearanceReconstruction() - this->warpedImage;
    this->errorImage = errorImage.reshape(1,1);
}

void AAM::calcErrorWeights() {
    this->errorWeights = this->calcWeights();
}

Mat AAM::calcWeights() {
    Mat i = this->errorImage.clone();

    float median = 0;
    Mat Input = abs(i.reshape(1,1)); // spread Input Mat to single row
    std::vector<float> vecFromMat;
    Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat

    int start = this->triangleMask.rows*this->triangleMask.cols-countNonZero(this->triangleMask);

    nth_element(vecFromMat.begin(), vecFromMat.begin()+(vecFromMat.size()-start)/2+start, vecFromMat.end());
    median = vecFromMat[(vecFromMat.size()-start)/2+start];

    //cout<<"Median: "<<median<<endl;

    float standardDeviation = 1.4826*(1+5/(Input.cols-start - this->s.rows))*median;
    //cout<<"Standard Deviation: "<<standardDeviation<<endl;

    this->outliers = Mat::zeros(i.rows, i.cols, CV_32FC1);

    //float stdPi = 1/(standardDeviation*sqrt(2*M_PI));
    //float stdSq = 2*standardDeviation*standardDeviation;

    for(int row=0; row<i.rows; row++) {
        for(int col=0; col<i.cols; col++) {
            float absValue = abs(i.fl(row, col));
            if(absValue < standardDeviation) {
                i.fl(row, col) = 1;
            } else if(absValue < 3*standardDeviation) {
                i.fl(row, col) = standardDeviation/absValue;
            } else {
                i.fl(row, col) = 0;
                this->outliers.fl(row, col) = 1.0f;
            }
            //i.fl(row, col) = (stdPi)*exp(-abs(i.fl(row, col))/(stdSq));
        }
    }

    /*
    namedWindow("WEIGHTS");
    imshow("WEIGHTS", i.reshape(1, this->modelHeight));
    */

    return i;
}


bool AAM::isInitialized() {
    return this->initialized;
}

bool AAM::hasFittingImage() {
    return !this->fittingImage.empty();
}
