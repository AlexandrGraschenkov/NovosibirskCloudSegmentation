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

namespace pcl_algo {
void processCloud(PointCloudRef cloud, std::vector<Type> &outTypes);

struct PointInfo {
    float groundHeight;
    float nearestPointDist;
    int nearestCount;
    int nearestCountSameRef;
    cv::Point3f meanOffset;
    std::array<cv::Point3f, 4> valAndDir;
    std::array<cv::Point3f, 4> valAndDirSameRef;
};

void generateFeatures(PointCloudRef cloud, std::vector<PointInfo> &outInfo, float radius = 0.6);
} // namespace pcl_algo

#endif /* cloud_analyzer_hpp */
