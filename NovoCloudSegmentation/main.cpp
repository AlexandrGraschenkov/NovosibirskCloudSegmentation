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
    
    string dataPath = "/Users/alex/Desktop/Новосибирск/test_dataset_test.csv";
    string predPath = "/Users/alex/Downloads/test_preds_hd200_elu_fixed.csv";
//    string predPath = "/Users/alex/Desktop/Новосибирск/test_pred_out/6fc_3bn/test_preds_24.csv";
//    string predPath = "/Users/alex/Desktop/Новосибирск/test_preds.csv";
//    string dataPath = "/Users/alex/Desktop/Новосибирск/train_dataset_train.csv";
//    string predPath = "/Users/alex/Desktop/Новосибирск/preds.csv";
//    string predPath = "/Users/alex/Desktop/Новосибирск/pred_classes.csv";
    PointCloudRef cloud = readCSV(dataPath);
    cloud->classes = readClassesCSV(predPath);
//    cout << "Filter start" << endl;
//    cloud->filterPoints();
//    cout << "Filter stop" << endl;
    cout << cloud->points.size() << endl;
    
    PangolinDSOViewer viewer(1280, 720);
    viewer.prepare();
//    viewer.updateCloud(cloud, 1);
    
    viewer.updateCloud(cloud, 1);
    viewer.origCloudPath = dataPath;
    viewer.savePredictionsPath = predPath+"_fixed";
//    viewer.predictionsPath = predPath;
    viewer.run();
    
    return 0;
}
