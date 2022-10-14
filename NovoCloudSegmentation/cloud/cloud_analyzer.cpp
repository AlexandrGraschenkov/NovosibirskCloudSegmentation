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
#include <pcl/common/geometry.h>
#include <unordered_set>
#include "plane.hpp"
#include <chrono>
#include <pcl/features/fpfh_omp.h>
#include <iomanip>
#include <iostream>
#include <unordered_map>

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

void connectedComponentsInRadius(PclTree &tree, int seedIdx, float radius, float stepRadius, std::vector<int> &outIdxs) {
    vector<int> indexes;
    vector<float> sqrDists;
    tree.radiusSearch(seedIdx, radius, indexes, sqrDists);
    // сохраняем какие индексы могут участвовать при поиске шагами
    // так по идее быстрей должно работать, чем после псотоянно сравнивать дистанцию
    const unordered_set<int> allowedIdxs(indexes.begin(), indexes.end());
    
    unordered_set<int> processedIdx;
    unordered_set<int> toProcessIdx = {seedIdx};
    
    while (toProcessIdx.size()) {
        auto begin = toProcessIdx.begin();
        int idx = *begin;
        toProcessIdx.erase(begin);
        processedIdx.insert(idx);
        
        tree.radiusSearch(idx, stepRadius, indexes, sqrDists);
        for (int i : indexes) {
            if (processedIdx.count(i)) continue;
            if (allowedIdxs.count(i) == 0) continue;
            
            toProcessIdx.insert(i);
        }
    }
    outIdxs = vector<int>(processedIdx.begin(), processedIdx.end());
}

std::unordered_map<Type, int> countTypes(const vector<int> &idxs, const vector<Type>  &types) {
    unordered_map<Type, int> result;
    for (int i : idxs) {
        result[types[i]]++;
    }
    return result;
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
float getLocalMedianGroundHeight(PointCloudRef cloud, const vector<int> &indexes) {
    vector<float> hArr;
    hArr.reserve(indexes.size());
    for (int i : indexes) {
        hArr.push_back(cloud->points[i].z);
    }
    
    size_t pos = (hArr.size() * 0.8);
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
    
    float medianGroundGlobal = medianGround + cloud->offset.z;
    cout << "Ground " << fixed << setprecision(5) << medianGroundGlobal << endl;
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
        if (isnan(v.planeCurvature) || isnan(v.planeNormal.x)) {
            v.planeCurvature = 0;
            v.planeNormal = Point3f();
        }
        
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


void generateFeatures2(PointCloudRef cloud, std::vector<pcl::FPFHSignature33> &outFeatures, float radius) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    
    cout << "--- Features 2 ---" << endl;
    cout << "1/3 Compute normals!" << endl;
    auto t1 = std::chrono::steady_clock::now();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(pclCloud);
    
    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    
    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    
    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (radius/3);
    
    // Compute the features
    ne.compute (*cloud_normals);
    
    int nanNormalsCount = 0;
    for (int i = 0; i < cloud_normals->size(); i++) {
      if (!pcl::isFinite<pcl::Normal>((*cloud_normals)[i])) {
          (*cloud_normals)[i] = pcl::Normal();
          nanNormalsCount++;
//        PCL_WARN("normals[%d] is not finite\n", i);
      }
    }
    cout << "Nan normals " << nanNormalsCount << endl;
    
    cout << "2/3 Compute features!" << endl;
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " sec" << std::endl;
    
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> pfh;
    pfh.setInputCloud (pclCloud);
    pfh.setInputNormals (cloud_normals);
    pfh.setSearchMethod (tree);
    
    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr pfhs(new pcl::PointCloud<pcl::FPFHSignature33> ());
    
    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    pfh.setRadiusSearch (radius);
    
    // Compute the features
    pfh.compute (*pfhs);
    cout << "3/3 Transfer features!" << endl;
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count() << " sec" << std::endl;
    
    outFeatures = vector<pcl::FPFHSignature33>(pfhs->begin(), pfhs->end());
    
    cout << "--- Done ---" << endl;
    auto t4 = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " sec" << std::endl;
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
    int fixedCount = 0;
    for (size_t i = 0; i < size ; i++) {
        if (i%1000000 == 0) cout << i << " / " << size << endl;
//        if (i%processEveryN != 0) continue;
//        if (cloud->classes[i] != TypeNoise) continue;
        kdtree.nearestKSearch((int)i, 5, tempNoiseIdxs, tempNoiseSqrDists);
        if (tempNoiseSqrDists[3] > 0.7) {
            // смотрим что ближайшие точки лежат далеко
            inOutTypes[i] = TypeNoise;
            fixedCount++;
        }
    }
    cout << "Noise fixed count " << fixedCount << endl;
}

struct Support {
    std::vector<int> indexes;
    cv::Vec3f size;
};

//void find

void fixSupports(PointCloudRef cloud, std::vector<Type> &inOutTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    
    unordered_set<int> usedIdxs;
//    for (int i = 0; i < )
}



void fixClasses(PointCloudRef cloud, std::vector<Type> &outTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    float medianGround = getMedianGroundHeight(cloud);
    
    outTypes = cloud->classes;
    const auto &inTypes = cloud->classes;
    vector<int> changeTypeCount = {0, 0, 0};
    const int size = (int)cloud->points.size();
    unordered_set<int> contactNetworkIdx;
    for (int i = 0; i < size; i++) {
        if (i%40000 == 0) {
            cout << i << " / " << size << endl;
            for (int val : changeTypeCount) cout << val << " ";
            cout << endl;
        }
        if (!(inTypes[i] == TypeGreenery || inTypes[i] == TypeNoise)) {
            continue;
        }
        vector<int> closestConnectedIndexes;
        connectedComponentsInRadius(kdtree, i, 1.0, 0.4, closestConnectedIndexes);
        if (closestConnectedIndexes.size() < 4) {
//            if (inTypes[i] != TypeNoise) {
//                outTypes[i] = TypeNoise;
//                changeTypeCount[0]++;
//            }
            continue;
        }
        
//        pca.setIndices(make_shared<vector<int>>(closestConnectedIndexes));
//        auto vals = pca.getEigenValues();
//        if (vals[0] / vals[1] > 4) {
//            // it's some line
//            auto vecDir = pca.getEigenVectors().col(0);
//            bool vertical = abs(vecDir.z()) > 0.8;
//            if (!vertical && inTypes[i] == TypeNoise) {
//                outTypes[i] = TypeContactNetwork;
//                changeTypeCount[1]++;
//                continue;
//            }
//        }
        
        auto types = countTypes(closestConnectedIndexes, inTypes);
        vector<int> counts;
        Type maxType = TypeUnknown; int maxTypeCount = 0;
        for (auto &t : types) {
            counts.push_back(t.second);
            if (maxTypeCount < t.second) {
                maxTypeCount = t.second;
                maxType = t.first;
            }
        }
        sort(counts.begin(), counts.end());
        if (counts.size() > 1 && maxType == TypeSupports && inTypes[i] == TypeGreenery) {
            outTypes[i] = maxType;
            changeTypeCount[1]++;
            continue;
        }
        float h = cloud->points[i].z - medianGround;
        if (counts.size() > 1 && maxType == TypeContactNetwork && maxType != inTypes[i] && h > 3) {
            
            for (int idx : closestConnectedIndexes) {
                if (inTypes[idx] == inTypes[i] && outTypes[idx] != TypeContactNetwork) {
                    contactNetworkIdx.insert(idx);
                }
            }
            contactNetworkIdx.insert(i);
        }
        changeTypeCount[2] = contactNetworkIdx.size();
    } 
    for (int idx : contactNetworkIdx) {
        outTypes[idx] = TypeContactNetwork;
    }
    cout << "Auto fix types count: ";
    for (int val : changeTypeCount) cout << val << " ";
    cout << endl;
}

void fixClasses2(PointCloudRef cloud, std::vector<Type> &outTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    PclPCA pca;
    pca.setInputCloud(pclCloud);
    float medianGround = getMedianGroundHeight(cloud);
    
    outTypes = cloud->classes;
    const auto &inTypes = cloud->classes;
    vector<int> changeTypeCount = {0, 0, 0};
    const int size = (int)cloud->points.size();
    unordered_set<int> contactNetworkIdx;
    for (int i = 0; i < size; i++) {
        if (i%40000 == 0) {
            cout << i << " / " << size << endl;
            for (int val : changeTypeCount) cout << val << " ";
            cout << endl;
        }
        const auto &p = cloud->points[i];
        float h = p.z - medianGround;
        if (inTypes[i] == TypeNoise && (p.x + p.y > -20) && h < 1) {
//            continue;
            // process
        } else {
            continue;
        }
        vector<int> closestConnectedIndexes;
        connectedComponentsInRadius(kdtree, i, 0.7, 0.2, closestConnectedIndexes);
        if (closestConnectedIndexes.size() < 10) {
//            if (inTypes[i] != TypeNoise) {
//                outTypes[i] = TypeNoise;
//                changeTypeCount[0]++;
//            }
            continue;
        }
        float localGround = getLocalMedianGroundHeight(cloud, closestConnectedIndexes);
        
        if (p.x + p.y > 20 && p.z > localGround) {
            outTypes[i] = TypeGreenery;
            changeTypeCount[0]++;
        } else {
            outTypes[i] = TypeGround;
            changeTypeCount[1]++;
        }
    }
}

} // namespace pcl_algo

