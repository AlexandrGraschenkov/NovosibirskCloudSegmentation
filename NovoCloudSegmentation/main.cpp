//
//  main.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 01.10.2022.
//

#include <iostream>
#include "cloud/cloud.hpp"
#include "3d_visualization/PangolinDSOViewer.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
//    string dataPath = "/Users/alex/Desktop/Новосибирск/test_dataset_test.csv";
    string dataPath = "/Users/alex/Desktop/Новосибирск/train_dataset_train.csv";
    string predPath = "/Users/alex/Desktop/Новосибирск/pred_classes.csv";
    PointCloudRef cloud = readCSV(dataPath);
    cout << cloud->points.size() << endl;
    
    PangolinDSOViewer viewer(1280, 720);
    viewer.prepare();
//    viewer.updateCloud(cloud, 1);
    viewer.updateCloud(cloud, 1);
    viewer.origCloudPath = dataPath;
    viewer.predictionsPath = predPath;
    viewer.run();
    
    return 0;
}
