//
//  SimpleMatrix.hpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 20.12.17.
//  Copyright © 2017 Alexander Graschenkov. All rights reserved.
//

#ifndef SimpleMatrix_hpp
#define SimpleMatrix_hpp

#include <stdio.h>
#include <math.h>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "../utils/sophus/sim3.hpp"
#include <pangolin/pangolin.h>



struct MyMatrix {
    Eigen::Affine3d m;
    
    MyMatrix();
    static MyMatrix sim3_vec(const Eigen::Matrix<double, 7, 1> &vec);
    static MyMatrix rotation(double x, double y, double z);
    static MyMatrix translation(double x, double y, double z);
    static MyMatrix scale(double x, double y, double z);
    static MyMatrix quanterion(double *q, double *pos);
    static MyMatrix identity();
    
    // val == 1 => result the same
    // val == 0.5 => do half motion, rotation and scale
    MyMatrix multiply(double val) const;
    MyMatrix apply(MyMatrix mat) const;
    cv::Point3d apply(const cv::Point3d &p) const;
    Eigen::Vector3d apply(const Eigen::Vector3d &p) const;
    MyMatrix inverse() const;
    pangolin::OpenGlMatrix toGL() const;
    
    bool isIdentity() const;
    
    double* data(); // column based data
    const double* data() const; // column based data
    
    cv::Point3d getTranslations() const;
    cv::Point3d getRotations() const;
    cv::Point3d getScale() const;
    
    MyMatrix interpolateTo(const MyMatrix &to, float percent);
    static std::vector<MyMatrix> interpolateMat(const MyMatrix &from, const MyMatrix &to, int count);
    static std::vector<cv::Point3f> interpolatePoints(const cv::Point3f &from, const cv::Point3f &to, int count);
};

#endif /* SimpleMatrix_hpp */
