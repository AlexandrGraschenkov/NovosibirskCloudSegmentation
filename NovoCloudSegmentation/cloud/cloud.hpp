//
//  cloud.hpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 01.10.2022.
//

#ifndef cloud_hpp
#define cloud_hpp

#include <stdio.h>
#include <opencv2/core.hpp>
#include <memory>

#define TYPE_COUNT 6

typedef enum : int {
    TypeUnknown = -1,
    TypeGround = 0,
    TypeSupports = 1,
    TypeGreenery = 3, // растительность
    TypeRails = 4,
    TypeContactNetwork = 5, // контактные сети
    TypeNoise = 64
} Type;

struct PointCloud {
    std::vector<int> ids;
    std::vector<cv::Point3f> points;
    std::vector<float> reflectance;
    std::vector<Type> classes;
    
    cv::Point3d offset;
    
    size_t size() const { return points.size(); }
};

// не дай боже копирнуть эту структурку
using PointCloudRef = std::shared_ptr<PointCloud>;

PointCloudRef readCSV(const std::string &fullPath, bool useCache = true);
std::vector<Type> readClassesCSV(const std::string &fullPath);
cv::Vec3f colorForType(Type t);

#endif /* cloud_hpp */
