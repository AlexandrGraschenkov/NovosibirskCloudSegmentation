//
//  SimpleMatrix.cpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 20.12.17.
//  Copyright © 2017 Alexander Graschenkov. All rights reserved.
//

#include "SimpleMatrix.hpp"


using namespace std;
using namespace cv;


MyMatrix MyMatrix::sim3_vec(const Eigen::Matrix<double, 7, 1> &vec) {
    Sophus::Sim3d sim = Sophus::Sim3d::exp(vec);
    
    MyMatrix result;
    result.m = Eigen::Affine3d(sim.matrix());
    return result;
}

MyMatrix MyMatrix::rotation(double x, double y, double z) {
    Eigen::Affine3d rx = Eigen::Affine3d(Eigen::AngleAxisd(x, Eigen::Vector3d(1, 0, 0)));
    Eigen::Affine3d ry = Eigen::Affine3d(Eigen::AngleAxisd(y, Eigen::Vector3d(0, 1, 0)));
    Eigen::Affine3d rz = Eigen::Affine3d(Eigen::AngleAxisd(z, Eigen::Vector3d(0, 0, 1)));
    MyMatrix m;
    m.m = rz * ry * rx;
    return m;
}
MyMatrix MyMatrix::translation(double x, double y, double z) {
    MyMatrix m;
    m.m = Eigen::Affine3d(Eigen::Translation3d(Eigen::Vector3d(x,y,z)));
    return m;
}
MyMatrix MyMatrix::scale(double x, double y, double z) {
    MyMatrix m;
    m.m = Eigen::Affine3d(Eigen::Scaling(x, y, z));
    return m;
}

MyMatrix MyMatrix::quanterion(double *q, double *pos) {
    MyMatrix m;
    m.m = Eigen::Affine3d(Eigen::Quaternion<double>(q[0], q[1], q[2], q[3]));
    m.data()[12] = pos[0];
    m.data()[13] = pos[1];
    m.data()[14] = pos[2];
    return m;
}
MyMatrix MyMatrix::identity() {
    return MyMatrix();
}

MyMatrix::MyMatrix() {
    m = Eigen::Affine3d::Identity();
}

MyMatrix MyMatrix::apply(MyMatrix mat) const {
    MyMatrix result;
    result.m = this->m * mat.m;
    return result;
}
Point3d MyMatrix::apply(const Point3d &p) const {
    Eigen::Vector3d vec(p.x,p.y,p.z);
    Eigen::Vector3d res = m * vec;
    return Point3d(res(0), res(1), res(2));
}
Eigen::Vector3d MyMatrix::apply(const Eigen::Vector3d &p) const {
    return m * p;
}
MyMatrix MyMatrix::inverse() const {
    MyMatrix result;
    result.m = m.inverse();
    return result;
}

MyMatrix MyMatrix::multiply(double val) const {
    Sophus::Sim3d sim(m.matrix());
    sim = Sophus::Sim3d::exp(sim.log() * val);
    
    MyMatrix result;
    result.m = Eigen::Affine3d(sim.matrix());
    return result;
}


bool MyMatrix::isIdentity() const {
    MyMatrix id = MyMatrix::identity();
    double *d1 = id.data();
    double *d2 = (double *)m.data();
    for (int i = 0; i < 16; i++) {
        if (d1[i] != d2[i]) {
            return false;
        }
    }
    return true;
}

double* MyMatrix::data() {
    return (double*)m.data();
}
const double* MyMatrix::data() const {
    return (double*)m.data();
}

Point3d setNearZeroToZero(Point3d p) {
    if (abs(p.x) < 0.0001)
        p.x = 0;
    if (abs(p.y) < 0.0001)
        p.y = 0;
    if (abs(p.z) < 0.0001)
        p.z = 0;
    return p;
}

Point3d MyMatrix::getTranslations() const {
    auto t = m.translation();
    Point3d p(t(0), t(1), t(2));
    return setNearZeroToZero(p);
}

Point3d MyMatrix::getRotations() const {
    Eigen::Vector3d ea = m.rotation().eulerAngles(0, 1, 2);
    Point3d p(ea(0), ea(1), ea(2));
    return setNearZeroToZero(p);
}

Point3d MyMatrix::getScale() const {
    Point3d vecX(m(0, 0), m(0, 1), m(0, 2));
    Point3d vecY(m(1, 0), m(1, 1), m(1, 2));
    Point3d vecZ(m(2, 0), m(2, 1), m(2, 2));
    return Point3d(norm(vecX), norm(vecY), norm(vecZ));
}


pangolin::OpenGlMatrix MyMatrix::toGL() const {
    pangolin::OpenGlMatrix m;
    const double *d = data();
    std::copy(d, d + 16, m.m);
    return m;
}

MyMatrix MyMatrix::interpolateTo(const MyMatrix &to, float percent) {
    MyMatrix mat = to.apply(this->inverse());
    return mat.multiply(percent).apply(*this);
}

std::vector<MyMatrix> MyMatrix::interpolateMat(const MyMatrix &from, const MyMatrix &to, int count) {
    MyMatrix mat = to.apply(from.inverse());
    std::vector<MyMatrix> result;
    result.reserve(count);
    for (int i = 1; i < count+1; i++) {
        float percent = i / (float)(count+1);
        result.push_back(mat.multiply(percent).apply(from));
    }
    return result;
}

std::vector<cv::Point3f> MyMatrix::interpolatePoints(const cv::Point3f &from, const cv::Point3f &to, int count) {
    std::vector<cv::Point3f> result;
    result.reserve(count);
    for (int i = 1; i < count+1; i++) {
        float percent = i / (float)(count+1);
        result.push_back((to - from) * percent + from);
    }
    return result;
}
