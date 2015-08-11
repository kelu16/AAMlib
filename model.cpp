#include "model.h"

Model::Model()
{
    this->initialized = false;
}

void Model::loadDataFromFile(string fileName) {
    TrainingData::loadDataFromFile(fileName);

    this->selected = Mat::zeros(this->groups.rows, this->groups.cols, CV_32FC1);
    this->calcTriangleStructure();
    this->initialized = true;
}

/*
void Model::setPoints(Mat points) {
    this->points = points;
    this->calcTriangleStructure();
}

void Model::setDescriptions(vector<string> descriptions) {
    this->descriptions = descriptions;
}
*/

Mat Model::getTriangles() {
    return this->triangles;
}

void Model::placeModelInBounds(Rect bounds) {
    Mat x = this->points.col(0);
    Mat y = this->points.col(1);

    double minX, minY, maxX, maxY;
    minMaxIdx(x, &minX, &maxX);
    minMaxIdx(y, &minY, &maxY);

    double scale = bounds.width/(maxX-minX);
    scale *= 0.8; //For best initial fit, depends on the used model

    //Center at (0,0)
    x = x-minX-(maxX-minX)/2;
    y = y-minY-(maxY-minY)/2;

    //Scale
    x *= scale;
    y *= scale;

    //Move to Position of found Face;
    x += bounds.x+bounds.width/2;
    y += bounds.y+bounds.height/2;
}

void Model::placeGroupInBounds(Rect bounds, int group, float scalingFactor) {
    Mat x = this->points.col(0);
    Mat y = this->points.col(1);

    double minX, minY, maxX, maxY;
    int n=x.rows;
    minMaxIdx(x, &maxX, &minX);
    minMaxIdx(y, &maxY, &minY);

    for(int i=0; i<n; i++) {
        if((int)this->groups.at<float>(i,0) & group) {
            minX = min(minX, (double)x.fl(i));
            maxX = max(maxX, (double)x.fl(i));
            minY = min(minY, (double)y.fl(i));
            maxY = max(maxY, (double)y.fl(i));
        }
    }

    for(int i=0; i<n; i++) {
        if((int)this->groups.at<float>(i,0) & group) {
            x.fl(i) = x.fl(i)-minX-(maxX-minX)/2;
            y.fl(i) = y.fl(i)-minY-(maxY-minY)/2;
        }
    }

    double scale = bounds.width/(maxX-minX);
    scale *= scalingFactor;

    for(int i=0; i<n; i++) {
        if((int)this->groups.at<float>(i,0) & group) {
            x.fl(i) *= scale;
            y.fl(i) *= scale;
        }
    }

    for(int i=0; i<n; i++) {
        if((int)this->groups.at<float>(i,0) & group) {
            x.fl(i) += bounds.x+bounds.width/2;
            y.fl(i) += bounds.y+bounds.height/2;
        }
    }
}

void Model::placeMouthInBounds(Rect bounds) {
    this->placeGroupInBounds(bounds, MODEL_MOUTH, 0.6);
}

void Model::placeLeftEyeInBounds(Rect bounds) {
    this->placeGroupInBounds(bounds, MODEL_LEFTEYE, 0.8);
}

void Model::placeRightEyeInBounds(Rect bounds) {
    this->placeGroupInBounds(bounds, MODEL_RIGHTEYE, 0.8);
}

void Model::placeNoseInBounds(Rect bounds) {
    this->placeGroupInBounds(bounds, MODEL_NOSE, 0.6);
}

void Model::scaleSelectedPoints(float scale) {
    this->unscaledPoints = this->points;

    Mat x = this->points.col(0);
    Mat y = this->points.col(1);

    double minX, minY, maxX, maxY;
    int n=x.rows;
    minMaxIdx(x, &maxX, &minX);
    minMaxIdx(y, &maxY, &minY);

    for(int i=0; i<n; i++) {
        if(this->selected.fl(i,0) == 1) {
            minX = min(minX, (double)x.fl(i));
            maxX = max(maxX, (double)x.fl(i));
            minY = min(minY, (double)y.fl(i));
            maxY = max(maxY, (double)y.fl(i));
        }
    }

    for(int i=0; i<n; i++) {
        if(this->selected.fl(i,0) == 1) {
            x.fl(i) = x.fl(i)-minX-(maxX-minX)/2;
            y.fl(i) = y.fl(i)-minY-(maxY-minY)/2;
        }
    }

    for(int i=0; i<n; i++) {
        if(this->selected.fl(i,0) == 1) {
            x.fl(i) *= scale;
            y.fl(i) *= scale;
        }
    }

    for(int i=0; i<n; i++) {
        if(this->selected.fl(i,0) == 1) {
            x.fl(i) += minX+(maxX-minX)/2;
            y.fl(i) += minY+(maxY-minY)/2;
        }
    }

    cout<<scale<<endl;
}

void Model::selectPoint(int id) {
    this->selected.at<float>(id,0) = 1;
}

void Model::selectPointsInRect(Rect selection) {
    this->unselectAllPoints();
    for(int i=0; i<this->points.rows; i++) {
        if((this->points.fl(i,0) >= selection.x) && (this->points.fl(i,0) <= (selection.x+selection.width))
        && (this->points.fl(i,1) >= selection.y) && (this->points.fl(i,1) <= (selection.y+selection.height))) {
            this->selectPoint(i);
        }
    }
}

void Model::unselectPoint(int id) {
    this->selected.at<float>(id,0) = 0;
}

void Model::unselectAllPoints() {
    int numPoints = this->selected.rows;
    for(int i=0; i<numPoints; i++) {
        this->unselectPoint(i);
    }
}

Point2f Model::getPointFromMat(Mat p, int id) {
    return Point2f(p.fl(id,0), p.fl(id,1));
}

Point2f Model::getPoint(int id) {
    return this->getPointFromMat(this->points, id);
}

int Model::findPointToPosition(Point p, int tolerance) {
    Point pos;
    for(int i=0; i<=this->points.rows; i++) {
        pos = Point((int)this->points.fl(i,0), (int)this->points.fl(i,1));
        if((abs(pos.x-(int)p.x) <= tolerance) && (abs(pos.y-(int)p.y) <= tolerance)) {
            return i;
        }
    }
    return -1;
}

int Model::findPointToPosition(Point p) {
    return this->findPointToPosition(p, 5);
}

void Model::moveSelectedVertices(float dx, float dy) {
    for(int i=0; i<=this->points.rows; i++) {
        if(this->selected.fl(i,0) == 1) {
            this->points.fl(i,0) -= dx;
            this->points.fl(i,1) -= dy;
        }
    }
}

vector <int> Model::getSelectedPoints() {
    vector <int> out;
    out.clear();
    for(int i=0; i<this->selected.rows; i++) {
        if(this->selected.fl(i,0) == 1) {
            out.push_back(i);
        }
    }

    return out;
}

bool Model::isPointSelected(int id) {
    return (this->selected.at<float>(id) == 1);
}

void Model::calcTriangleStructure() {
    Mat points_backup = this->points.clone();
    this->placeModelInBounds(Rect(100,100,300,300));
    vector<Vec6f> triangleList;
    bool insert;
    Subdiv2D subdiv;
    int counter = 0;

    double minX, minY, maxX, maxY;
    minMaxIdx(this->points.col(0), &minX, &maxX);
    minMaxIdx(this->points.col(1), &minY, &maxY);

    subdiv.initDelaunay(Rect(0,0,(10*maxX+10),(10*maxY+10)));

    for(int i=0; i<this->points.rows; i++) {
        Point v = this->getPoint(i);
        subdiv.insert(v);
    }

    subdiv.getTriangleList(triangleList);

    this->triangles = Mat::zeros(triangleList.size(), 3, CV_32S);

    for(unsigned int i=0; i<triangleList.size(); i++) {
        Vec6f t = triangleList[i];
        vector<Point2f> pt(3);

        pt[0] = Point(t[0], t[1]);
        pt[1] = Point(t[2], t[3]);
        pt[2] = Point(t[4], t[5]);

        insert = true;

        for(int j=0; j<3; j++) {
            if(pt[j].x > maxX+10 || pt[j].y > maxY+10 || pt[j].x < 0 || pt[j].y < 0) {
                insert = false;
                break;
            }
        }

        if(insert) {
            if(pt[0]!=pt[1] && pt[0]!=pt[2] && pt[1]!=pt[2]) {
                int posA, posB, posC;
                posA = this->findPointToPosition(pt[0], 1);
                posB = this->findPointToPosition(pt[1], 1);
                posC = this->findPointToPosition(pt[2], 1);

                this->triangles.at<int>(counter, 0) = posA;
                this->triangles.at<int>(counter, 1) = posB;
                this->triangles.at<int>(counter, 2) = posC;
                counter++;
            }
        }
    }

    this->triangles = this->triangles(cv::Rect(0,0,3,counter));
    this->points = points_backup;
}

bool Model::isInitialized() {
    return this->initialized;
}
