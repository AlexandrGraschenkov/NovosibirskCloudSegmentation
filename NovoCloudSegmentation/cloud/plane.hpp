//
//  plane.hpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 06.10.2022.
//

#ifndef plane_hpp
#define plane_hpp

#include <stdio.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core.hpp>

namespace pcl_algo {
struct Plane {
    cv::Point3f center;
    cv::Point3f normal;
    float curvature;
    
    static Plane findPlaneNE(const pcl::PointCloud<pcl::PointXYZ> &cloud, const std::vector<int> &indexes);
    
    float getPointDist(const cv::Point3f &p);
    cv::Point3f getPointNormalDist(const cv::Point3f &p);
};
} // namespace pcl_algo
#endif /* plane_hpp */
