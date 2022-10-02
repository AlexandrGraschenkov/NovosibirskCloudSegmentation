//
//  main.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 01.10.2022.
//

#include <iostream>
#include "cloud/cloud.hpp"
#include "PangolinDSOViewer.h"

using namespace std;
using namespace cv;

void start3d(PointCloudRef cloud) {
    PangolinDSOViewer viewer(1280, 720);
    viewer.prepare();
//    viewer.updateCloud(cloud, 1);
    viewer.updateCloud(cloud, 1);
    viewer.run();
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
    PointCloudRef cloud = readCSV("/Users/alex/Desktop/Новосибирск/train_dataset_train.csv");
    cout << cloud->points.size() << endl;
    start3d(cloud);
    
    return 0;
}
