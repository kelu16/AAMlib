#ifndef CVIMAGEWIDGET_H
#define CVIMAGEWIDGET_H

#endif // CVIMAGEWIDGET_H
#pragma once
#include <QWidget>
#include <QImage>
#include <QPainter>
#include <opencv2/opencv.hpp>

class CVImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit CVImageWidget(QWidget *parent = 0);

    QSize sizeHint() const { return _qimage.size(); }
    QSize minimumSizeHint() const { return _qimage.size(); }

signals:
    void clicked(QMouseEvent *ev, QPoint pos);
    void wheel(QWheelEvent *ev);

public slots:
    void showImage(const cv::Mat& image, bool resize=true);

protected:
    void paintEvent(QPaintEvent* /*event*/);
    void mousePressEvent(QMouseEvent *ev);
    void mouseReleaseEvent(QMouseEvent *ev);
    void mouseMoveEvent(QMouseEvent *ev);
    void wheelEvent(QWheelEvent *ev);


    QImage _qimage;
    cv::Mat _tmp;
};


