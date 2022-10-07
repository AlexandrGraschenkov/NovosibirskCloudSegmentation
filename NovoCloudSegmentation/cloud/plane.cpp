//
//  plane.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 06.10.2022.
//

#include "plane.hpp"
//#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

using namespace std;
using namespace cv;

namespace pcl_algo {

float calculateMedian(vector<float> vec) { // копируем вектор, т.к. он меняется
    if (vec.size() == 0) return 0;
    int pos = (int)vec.size()/2;
    nth_element(vec.begin(), vec.begin() + pos, vec.end());
    return vec[pos];
}

pcl::PointXYZ getMedianPoint(const pcl::PointXYZ *points, const vector<int> &indexes) {
    vector<float> xx, yy, zz;
    xx.reserve(indexes.size());
    yy.reserve(indexes.size());
    zz.reserve(indexes.size());
    for (int i : indexes) {
        xx.push_back(points[i].x);
        yy.push_back(points[i].y);
        zz.push_back(points[i].z);
    }
    float mx = calculateMedian(xx);
    float my = calculateMedian(yy);
    float mz = calculateMedian(zz);
    return pcl::PointXYZ(mx, my, mz);
}

Plane Plane::findPlaneNE(const pcl::PointCloud<pcl::PointXYZ> &cloud, const std::vector<int> &indexes) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointXYZ normalP;
    float curvature = 0;
    ne.computePointNormal(cloud, indexes, normalP.x, normalP.y, normalP.z, curvature);
    auto center = getMedianPoint(cloud.data(), indexes);
    Plane plane;
    plane.normal = Point3f(normalP.x, normalP.y, normalP.z);
    plane.center = Point3f(center.x, center.y, center.z);
    plane.curvature = curvature;
    return plane;
}

float Plane::getPointDist(const cv::Point3f &p) {
    // https://stackoverflow.com/a/3863777/820795
    return normal.dot(p - center);
}

cv::Point3f Plane::getPointNormalDist(const cv::Point3f &p) {
    float dist = getPointDist(p);
    return normal * dist;
}
} // namespace pcl_algo
