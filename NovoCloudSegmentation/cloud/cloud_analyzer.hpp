//
//  cloud_analyzer.hpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 02.10.2022.
//

#ifndef cloud_analyzer_hpp
#define cloud_analyzer_hpp

#include <stdio.h>
#include "cloud.hpp"
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>

namespace pcl_algo {
void processCloud(PointCloudRef cloud, std::vector<Type> &outTypes);

struct PointInfo {
    float groundHeight;
    float nearestPointDist[3];
    int nearestCount;
    int nearestCountSameRef;
    cv::Point3f meanOffset;
    cv::Point3f meanOffsetSmall;
    std::array<cv::Point3f, 4> valAndDir;
    std::array<cv::Point3f, 4> valAndDirSameRef;
    std::array<cv::Point3f, 4> valAndDirSmall;
    
    cv::Point3f planeNormal;
    float planeCurvature;
};

void generateFeatures(PointCloudRef cloud, std::vector<PointInfo> &outInfo, float radius = 0.6);
void generateFeatures2(PointCloudRef cloud, std::vector<pcl::FPFHSignature33> &outFeatures, float radius = 0.5);


void fixNoise(PointCloudRef cloud, std::vector<Type> &inOutTypes);
} // namespace pcl_algo

#endif /* cloud_analyzer_hpp */
