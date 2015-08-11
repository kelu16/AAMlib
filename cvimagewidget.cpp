#include "cvimagewidget.h"
#include <QMouseEvent>

using namespace std;

CVImageWidget::CVImageWidget(QWidget *parent) :
    QWidget(parent)
{
    setMouseTracking(true);
    this->setFixedSize(this->maximumSize());
}

void CVImageWidget::showImage(const cv::Mat& image, bool resize) {
    // Convert the image to the RGB888 format
    switch (image.type()) {
    case CV_32FC1:
        image.convertTo(_tmp, CV_8UC1, 255);
        cvtColor(_tmp, _tmp, CV_GRAY2RGB);
        break;
    case CV_32FC3:
        image.convertTo(_tmp, CV_8UC3, 255);
        cvtColor(_tmp, _tmp, CV_BGR2RGB);
        break;
    case CV_8UC1:
        cvtColor(image, _tmp, CV_GRAY2RGB);
        break;
    case CV_8UC3:
        cvtColor(image, _tmp, CV_BGR2RGB);
        break;
    }

    // QImage needs the data to be stored continuously in memory
    assert(_tmp.isContinuous());
    // Assign OpenCV's image buffer to the QImage. Note that the bytesPerLine parameter
    // (http://qt-project.org/doc/qt-4.8/qimage.html#QImage-6) is 3*width because each pixel
    // has three bytes.
    _qimage = QImage(_tmp.data, _tmp.cols, _tmp.rows, _tmp.cols*3, QImage::Format_RGB888);

    if(resize) {
        this->setFixedSize(image.cols, image.rows);
    }

    repaint();
}

void CVImageWidget::paintEvent(QPaintEvent *) {
    // Display the image
    QPainter painter(this);
    painter.drawImage(QPoint(0,0), _qimage);
    painter.end();
}

void CVImageWidget::mousePressEvent(QMouseEvent *ev) {
    QPoint pos = this->mapFromGlobal(QCursor::pos());
    emit clicked(ev, pos);
}

void CVImageWidget::mouseReleaseEvent(QMouseEvent *ev) {
    QPoint pos = this->mapFromGlobal(QCursor::pos());
    emit clicked(ev, pos);
}

void CVImageWidget::mouseMoveEvent(QMouseEvent *ev) {
    QPoint pos = this->mapFromGlobal(QCursor::pos());
    emit clicked(ev, pos);
}

void CVImageWidget::wheelEvent(QWheelEvent *ev) {
    emit wheel(ev);
}
