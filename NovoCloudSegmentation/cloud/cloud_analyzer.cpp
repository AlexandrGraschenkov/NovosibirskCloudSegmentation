//
//  cloud_analyzer.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 02.10.2022.
//

#include "cloud_analyzer.hpp"
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/pca.h>
#include <unordered_set>
#include "plane.hpp"

using namespace std;
using namespace cv;

using PclCloud = pcl::PointCloud<pcl::PointXYZ>;
using PclTree = pcl::KdTreeFLANN<pcl::PointXYZ>;
using PclPCA = pcl::PCA<pcl::PointXYZ>;

namespace pcl_algo {


// конвертация точек в формат PCL
PclCloud::Ptr makeCloud(const std::vector<cv::Point3f> &cloud) {
    PclCloud::Ptr pclCloud(new PclCloud);
    int count = (int)(cloud.size());
    
    pclCloud->width = count;
    pclCloud->height = 1;
    pclCloud->points.resize(pclCloud->width * pclCloud->height);
    auto &points = pclCloud->points;
    for (int i = 0; i < count; i++) {
        points[i] = pcl::PointXYZ(cloud[i].x,
                                  cloud[i].y,
                                  cloud[i].z);
    }
    return pclCloud;
}

void filterIndexes(const vector<float> &reflectance, float minRef, float maxRef, const vector<int> &origIdxs, vector<int> &outIdxs) {
    outIdxs.clear();
    for (const int &i : origIdxs) {
        if (minRef <= reflectance[i] && reflectance[i] < maxRef) {
            outIdxs.push_back(i);
        }
    }
}

inline cv::Point3f eigenToCv(const Eigen::Vector3f &vec) {
    return Point3f(vec[0], vec[1], vec[2]);
}
inline cv::Point3f eigen4ToCv(const Eigen::Vector4f &vec) {
    return Point3f(vec[0], vec[1], vec[2]);
}

std::array<cv::Point3f, 4> getPCAFeature(PclPCA &pca) {
    std::array<cv::Point3f, 4> res;
    res[0] = eigenToCv(pca.getEigenValues());
    res[1] = eigenToCv(pca.getEigenVectors().col(0));
    res[2] = eigenToCv(pca.getEigenVectors().col(1));
    res[3] = eigenToCv(pca.getEigenVectors().col(2));
    return res;
}

float getMedianGroundHeight(PointCloudRef cloud) {
    vector<float> hArr(cloud->points.size());
    for (size_t i = 0; i < hArr.size(); i++) {
        hArr[i] = cloud->points[i].z;
    }
    
    size_t pos = (hArr.size() * 0.9);
    nth_element(hArr.begin(), hArr.begin() + pos, hArr.end());
    return hArr[pos];
}

void generateFeatures(PointCloudRef cloud, std::vector<PointInfo> &outInfo, float radius) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    
    size_t size = cloud->points.size();
    outInfo.resize(size);
    float medianGround = getMedianGroundHeight(cloud);
    
    const vector<float> &reflectance = cloud->reflectance;
    vector<float> tempNoiseSqrDists, tempSqrDists;
    vector<int> tempNoiseIdxs, tempIdxs, tempIdxs2;
    const bool smallData = true;
    const float smallRadius = radius / 3.0f;
    for (size_t i = 0; i < size ; i++) {
        if (i%10000 == 0) cout << i << " / " << size << endl;
        
        const auto &p = cloud->points[i];
        auto &v = outInfo[i];
        kdtree.nearestKSearch((int)i, 5, tempNoiseIdxs, tempNoiseSqrDists);
        
        // small radius
        if (smallData) {
            kdtree.radiusSearch((int)i, smallRadius, tempIdxs, tempSqrDists);
            pca.setIndices(make_shared<vector<int>>(tempIdxs));
        }
        if (smallData && tempIdxs.size() > 3) {
            v.valAndDirSmall = getPCAFeature(pca);
            v.meanOffsetSmall = eigen4ToCv(pca.getMean()) - p;
        } else {
            v.valAndDirSmall = {Point3f(), Point3f(), Point3f(), Point3f()};
            v.meanOffsetSmall = {};
        }
        
        // расчитываем плоскость для маленького радиуса
        Plane plane = Plane::findPlaneNE(*pclCloud, tempIdxs);
        v.planeNormal = plane.getPointNormalDist(p);
        v.planeCurvature = plane.curvature;
        
        kdtree.radiusSearch((int)i, radius, tempIdxs, tempSqrDists);
        pca.setIndices(make_shared<vector<int>>(tempIdxs));
        
        v.groundHeight = p.z - medianGround;
        v.nearestPointDist[0] = sqrt(tempNoiseSqrDists[1]);
        v.nearestPointDist[1] = sqrt(tempNoiseSqrDists[2]);
        v.nearestPointDist[2] = sqrt(tempNoiseSqrDists[3]);
        v.nearestCount = (int)tempIdxs.size();
        
        if (tempIdxs.size() > 3) {
            v.valAndDir = getPCAFeature(pca);
            v.meanOffset = eigen4ToCv(pca.getMean()) - p;
        } else {
            v.valAndDir = {Point3f(), Point3f(), Point3f(), Point3f()};
            v.meanOffset = {};
        }
        
        // для той же отражающей способности
        float ref = reflectance[i];
        filterIndexes(reflectance, ref-3, ref+3, tempIdxs, tempIdxs2);
        pca.setIndices(make_shared<vector<int>>(tempIdxs2));
        
        if (tempIdxs2.size() > 3) {
            v.valAndDirSameRef = getPCAFeature(pca);
        } else {
            v.valAndDirSameRef = {Point3f(), Point3f(), Point3f(), Point3f()};
        }
        v.nearestCountSameRef = (int)tempIdxs2.size();
    }
}

void processCloud(PointCloudRef cloud, std::vector<Type> &outTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    
    size_t size = cloud->points.size();
    outTypes.resize(size);
    
    vector<float> tempNoiseSqrDists, tempSqrDists;
    vector<int> tempNoiseIdxs, tempIdxs;
    int processEveryN = 10;
    const auto &inTypes = cloud->classes;
    for (size_t i = 0; i < size ; i++) {
        if (i%10000 == 0) cout << i << " / " << size << endl;
        outTypes[i] = TypeUnknown;
//        if (i%processEveryN != 0) continue;
//        if (cloud->classes[i] != TypeNoise) continue;
        kdtree.nearestKSearch((int)i, 3, tempNoiseIdxs, tempNoiseSqrDists);
        if (tempNoiseSqrDists[1] > 0.1) {
            // смотрим что ближайшие точки лежат далеко
//            if (inType[i] == TypeGround) {
//                outTypes[i] = TypeNoise;
//            }
            outTypes[i] = TypeNoise;
            continue;
        }
        
        kdtree.radiusSearch((int)i, 0.6, tempIdxs, tempSqrDists);
        if (tempIdxs.size() < 8) {
//            outTypes[i] = TypeNoise;
            continue;
        }
        
        pca.setIndices(make_shared<vector<int>>(tempIdxs));
        auto vals = pca.getEigenValues();
        if (vals[0] / vals[1] > 4) {
            // it's some line
            auto vecDir = pca.getEigenVectors().col(0);
            bool vertical = abs(vecDir.z()) > 0.8;
            if (vertical) {
                outTypes[i] = TypeSupports;
                auto mean = pca.getMean();
                Point3f p = cloud->points[i];
                Point3f meanOffset(mean[0]-p.x, mean[1]-p.y, mean[2]-p.z);
//                cout << cloud->points[i] << " <> " << pca.getMean() << endl;
//                if (inTypes[i] == TypeSupports) {
//                    cout << "•true " << norm(meanOffset) << endl;
////                    cout << "•true " << vals[0] << " " << vals[1] << " " << vals[2] << endl;
//                } else if (inTypes[i] == TypeContactNetwork) {
//                    cout << "false " << norm(meanOffset) << endl;
////                    cout << "false " << vals[0] << " " << vals[1] << " " << vals[2] << endl;
//                }
            } else {
                outTypes[i] = TypeContactNetwork;
            }
            continue;
        }
    }
}

void fixNoise(PointCloudRef cloud, std::vector<Type> &inOutTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    
    const size_t size = cloud->points.size();
    
    vector<float> tempNoiseSqrDists, tempSqrDists;
    vector<int> tempNoiseIdxs, tempIdxs;
    for (size_t i = 0; i < size ; i++) {
        if (i%10000 == 0) cout << i << " / " << size << endl;
//        if (i%processEveryN != 0) continue;
//        if (cloud->classes[i] != TypeNoise) continue;
        kdtree.nearestKSearch((int)i, 3, tempNoiseIdxs, tempNoiseSqrDists);
        if (tempNoiseSqrDists[1] > 0.1) {
            // смотрим что ближайшие точки лежат далеко
            inOutTypes[i] = TypeNoise;
        }
    }
}
} // namespace pcl_algo
