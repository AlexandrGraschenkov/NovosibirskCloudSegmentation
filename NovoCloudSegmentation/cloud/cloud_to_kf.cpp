//
//  cloud_to_kf.cpp
//  CellTowerAnalyzer
//
//  Created by Alexander Graschenkov on 09.06.2022.
//

#include "cloud_to_kf.hpp"
using namespace std;
using namespace cv;


void generateKeyframes(PointCloudRef cloud,
                       std::unordered_map<Type,
                       KeyFrameDisplay *> &kfMap,
                       float keepPercentSize,
                       std::vector<Type> *overrideTypes)
{
    int count = keepPercentSize * cloud->size();
    vector<Point3f> points(count);
    vector<Vec3f> colors(count);
    vector<Vec3f> segColors(count);
    
    float pointStep = 1 / keepPercentSize;
    float minRef = -40;
    float maxRef = 5;
    
    const auto &cloudPoints = cloud->points;
    for (size_t i = 0; i < points.size(); i++) {
        size_t ii = i * pointStep;
        
        Type t = TypeUnknown;
        if (ii < cloud->classes.size()) t = cloud->classes[ii];
        if (overrideTypes && ii < overrideTypes->size()) t = (*overrideTypes)[ii];
        if (kfMap.count(t) == 0) {
            kfMap[t] = new KeyFrameDisplay();
        }
        kfMap[t]->cloud.push_back(cloudPoints[ii]);
        float color = (cloud->reflectance[ii] - minRef) / (maxRef - minRef);
        color = MIN(1, MAX(0, color));
        kfMap[t]->colors.push_back(Vec3f(color, color, 1));
    }
    for (auto kv : kfMap) {
        kv.second->pushDataToBuffers();
    }
}
