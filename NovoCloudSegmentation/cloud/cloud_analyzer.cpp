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
#include <unordered_set>

using namespace std;
using namespace cv;

using PclCloud = pcl::PointCloud<pcl::PointXYZ>;
using PclTree = pcl::KdTreeFLANN<pcl::PointXYZ>;

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

void processCloud(PointCloudRef cloud, std::vector<Type> &outTypes) {
    PclCloud::Ptr pclCloud = makeCloud(cloud->points);
    PclTree kdtree;
    kdtree.setInputCloud(pclCloud);
    
    size_t size = cloud->points.size();
    outTypes.resize(size);
    
    vector<float> tempSqrDists;
    vector<int> tempIdxs;
    for (size_t i = 0; i < size ; i++) {
        if (cloud->classes[i] != TypeNoise) continue;
        kdtree.nearestKSearch(i, 5, tempIdxs, tempSqrDists);
        for (int ii = 0; ii < tempSqrDists.size(); ii++) {
            cout << sqrt(tempSqrDists[ii]) << " ";
        }
        cout << endl;
    }
}
} // namespace pcl_algo
