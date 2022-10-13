#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"
#include "MatrixChnageHandler.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include "../cloud/cloud_to_kf.hpp"
#include <pangolin/display/default_font.h>
#include <iomanip>
#include <iostream>
#include "../utils/confusion_matrix.hpp"
#include "../cloud/cloud_analyzer.hpp"


using namespace std;
using namespace cv;


void PangolinDSOViewer::saveParametrsToFile() {
//    ofstream file("save_params.txt");
//    file << maxDistance << maxRelBS << endl;
////    Matrix m = kbHandler->tranform;
////    double *data = m.data();
////    for (int i = 0; i < 16; i++) {
////        file << data[i] << " ";
////    }
//    file.close();
}
void PangolinDSOViewer::loadParametrsToFile() {
//    maxDistance = 90;
//    maxRelBS = 0.5;
//    ifstream file("save_params.txt");
//    if (!file.is_open()) return;
//
//    file >> maxDistance >> maxRelBS;
//    file.close();
//    for (auto fr : tracks) {
//        fr->setMaxDistance(maxDistance);
//    }
}

template <class T>
bool checkUpdate(T &a, const T&b) {
    if (a != b) {
        a = b;
        return true;
    }
    return false;
}


PangolinDSOViewer::PangolinDSOViewer(int w, int h)
{
    this->w = w;
    this->h = h;
    
    running = true;
}


void PangolinDSOViewer::updateCloud(PointCloudRef cloud, float keepPercentSize) {
    lock_guard<mutex> lock(framesMtx);
    generateKeyframes(cloud, kfPoints, keepPercentSize);
    this->cloud = cloud;
}

PangolinDSOViewer::~PangolinDSOViewer()
{
    close();
//    runThread.join();
}

void PangolinDSOViewer::join()
{
//    runThread.join();
    printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
    // todo
}

void PangolinDSOViewer::prepare() {
    printf("START PANGOLIN!\n");
    pangolin::CreateWindowAndBind("Main",w,h);
    glEnable(GL_DEPTH_TEST);
}

#pragma region - Draw


void PangolinDSOViewer::drawAll(float ptSize) {
    lock_guard<mutex> lock(framesMtx);
    
    auto keyframes = (settings_displayProcessed && kfPointsProcessed.size() > 0) ? &kfPointsProcessed : &kfPoints;
    for (const auto &kv : *keyframes) {
        auto color = colorForType(kv.first);
        float *colorPtr = settings_displaySegmentation ? &color[0] : NULL;
        kv.second->drawPC(ptSize, colorPtr);
    }
    if (errorPointsKf) {
        float color[3] = {1, 0.2, 0.2};
        errorPointsKf->drawPC(settings_errorPointsSize, color);
    }
    
    glLineWidth(10);
    pangolin::glDrawAxis(2);
}

#pragma MARK -
void PangolinDSOViewer::resetCameraPosition(pangolin::OpenGlRenderState &camera) {
    auto &points = cloud->points;
    cv::Mat pointsMat((int)points.size(), 3, CV_32F, points.data());
    cv::Point3d minP, maxP;
    cv::minMaxLoc(pointsMat.col(0), &minP.x, &maxP.x);
    cv::minMaxLoc(pointsMat.col(1), &minP.y, &maxP.y);
    cv::minMaxLoc(pointsMat.col(2), &minP.z, &maxP.z);
    
    cv::Point3d center = (minP + maxP) / 2.0, border = maxP;
    
    cv::Point3d offset(0, -5, -10);
    offset = center + offset / norm(offset) * norm(border-center) * 0.5;
//    cout << "Total Points " << pointsCount << endl;
//    cout << "Total Frames " << frCount << endl;
    cout << offset << center << border << endl;
    
    camera.SetProjectionMatrix(pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000000000000));
    camera.SetModelViewMatrix(pangolin::ModelViewLookAt(offset.x,offset.y,offset.z, center.x,center.y,center.z, pangolin::AxisNegY));
}

inline void MyReadFramebuffer(pangolin::TypedImage &buffer, const pangolin::Viewport& v, const std::string& pixel_format)
{
    const pangolin::PixelFormat fmt = pangolin::PixelFormatFromString(pixel_format);
    const pangolin::GlPixFormat glfmt(fmt);

    auto pitch = v.w*fmt.bpp/8;
    if (buffer.w != v.w || buffer.h != v.h || buffer.pitch != pitch) {
        buffer.Alloc(v.w, v.h, pitch);
    }
    
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(v.l, v.b, v.w, v.h, glfmt.glformat, glfmt.gltype, buffer.ptr );
}

void PangolinDSOViewer::run()
{
//    const int UI_WIDTH = 160;
    const int UI_WIDTH = 20* pangolin::default_font().MaxWidth();

    
    // 3D visualization
    pangolin::OpenGlRenderState Visualization3D_camera;

    kbHandler = new MatrixChnageHandler(Visualization3D_camera);
    resetCameraPosition(Visualization3D_camera);
    kbHandler->loadToFile("save_transform_matrix.txt");
    pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
        .SetHandler(kbHandler);
    loadParametrsToFile();

    // parameter reconfigure gui
    pangolin::View &panel = pangolin::CreatePanel("ui");
    panel.SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    
    pangolin::Var<bool> settings_resetCamPosition("ui.Reset Camera",false,false);
    pangolin::Var<float> settings_pointSize("ui.Point size", 3, 0.1, 10, false);
    
    pangolin::Var<bool> settings_processCloud("ui.Process cloud",false,false);
    pangolin::Var<bool> settings_loadPredictions("ui.Load predictions",false,false);
    pangolin::Var<bool> settings_generateFeatures("ui.Generate features",false,false);
    pangolin::Var<bool> settings_generateFeatures2("ui.Generate features 2",false,false);
    pangolin::Var<bool> settings_saveFeatures("ui.Save features",false,false);
    pangolin::Var<bool> settings_saveFeatures_2("ui.Save features 2",false,false);
    
    pangolin::Var<bool> settings_showSegmentation("ui.Segmentation",false,true);
    pangolin::Var<bool> settings_showProcessed("ui.Processed",false,true);
    pangolin::Var<float> settings_accuracy("ui.Accuracy", 0, false);
    
    pangolin::Var<float> settings_errorPointSize("ui.Error Point size", 3, 0.1, 10, false);
    pangolin::Var<bool> settings_fixNoiseButt("ui.Fix noise",false,false);
    
    pangolin::Var<bool> settings_rotateGenerateFeatures("ui.Rotate Generate Features",false,false);
    


//    int count = 0;
//    bool prepared = false;
    bool isFirstLoop = true;
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() && running )
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        {
            // Activate efficiently by object
            Visualization3D_display.Activate(Visualization3D_camera);
            drawAll(settings_pointSize.Get());
        }

        // update parameters
        settings_displaySegmentation = settings_showSegmentation.Get();
        settings_displayProcessed = settings_showProcessed.Get();
        this->settings_errorPointsSize = settings_errorPointSize.Get();
        bool changes = false;
        
        if (pangolin::Pushed(settings_resetCamPosition)) {
            resetCameraPosition(Visualization3D_camera);
        }
        if (pangolin::Pushed(settings_processCloud)) {
            runProcessCloud();
            settings_showProcessed = settings_displayProcessed;
            settings_accuracy = lastAcc;
        }
        if (pangolin::Pushed(settings_loadPredictions)) {
            loadPredictions();
            settings_showProcessed = settings_displayProcessed;
            settings_accuracy = lastAcc;
        }
        if (pangolin::Pushed(settings_fixNoiseButt)) {
            fixNoise();
            settings_accuracy = lastAcc;
        }
        if (pangolin::Pushed(settings_generateFeatures)) {
            generateFeatures();
        }
        if (pangolin::Pushed(settings_generateFeatures2)) {
            generateFeatures2();
        }
        if (pangolin::Pushed(settings_saveFeatures)) {
            saveFeatures();
        }
        if (pangolin::Pushed(settings_saveFeatures_2)) {
            saveFeatures2();
        }
        if (pangolin::Pushed(settings_rotateGenerateFeatures)) {
            rotateGenerateAndSaveFeatures();
        }
        
        if (changes) {
            saveParametrsToFile();
        }
        
        if (kbHandler->checkChanges()) {
            kbHandler->saveToFile("save_transform_matrix.txt");
        }

        // Swap frames and Process Events
        pangolin::FinishFrame();
        isFirstLoop = false;
        
    }


    printf("QUIT Pangolin thread!\n");
    printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");

    exit(1);
}

void PangolinDSOViewer::close()
{
    running = false;
}

// MARK: Cloud analyze algorithms

void PangolinDSOViewer::runProcessCloud() {
    vector<Type> newTypes;
    pcl_algo::processCloud(cloud, newTypes);
    updatePredictions();
}


void PangolinDSOViewer::generateFeatures() {
    pcl_algo::generateFeatures(cloud, cloudFeatures);
}

void PangolinDSOViewer::generateFeatures2() {
    pcl_algo::generateFeatures2(cloud, cloudFeatures2);
}

inline void writePoint(ofstream &file, const Point3f &p, bool endComma = true) {
    file << p.x << ",";
    file << p.y << ",";
    file << p.z;
    if (endComma) file << ",";
}

template <typename T>
void appendVector(vector<T> &vec, const vector<T> &appendVec) {
    vec.insert(vec.end(), appendVec.begin(), appendVec.end());
}

void PangolinDSOViewer::saveFeatures() {
    ofstream file(origCloudPath + "_result");
    file << fixed << setprecision(4);
    bool withClasses = !cloud->classes.empty();
    vector<string> rows = {"id", "Easting", "Northing", "Height", "Reflectance"};
    if (withClasses) {
        rows.push_back("Class");
    }
    
    appendVector(rows, {
        "GroundHeight", "NearestDist_0", "NearestDist_1", "NearestDist_2", "NearestCount", "NearestSameRefCount",
        
        "PlaneNormal.X", "PlaneNormal.Y", "PlaneNormal.Z",
        "PlaneCurvature", // 16
        
        "SMeanOffset.X", "SMeanOffset.Y", "SMeanOffset.Z",
        "SPCA_Mag.X", "SPCA_Mag.Y", "SPCA_Mag.Z",
        "SPCA_Vec0.X", "SPCA_Vec0.Y", "SPCA_Vec0.Z",
        "SPCA_Vec1.X", "SPCA_Vec1.Y", "SPCA_Vec1.Z",
        "SPCA_Vec2.X", "SPCA_Vec2.Y", "SPCA_Vec2.Z", // 31
        
        "MeanOffset.X", "MeanOffset.Y", "MeanOffset.Z",
        "PCA_Mag.X", "PCA_Mag.Y", "PCA_Mag.Z",
        "PCA_Vec0.X", "PCA_Vec0.Y", "PCA_Vec0.Z",
        "PCA_Vec1.X", "PCA_Vec1.Y", "PCA_Vec1.Z",
        "PCA_Vec2.X", "PCA_Vec2.Y", "PCA_Vec2.Z", // 46
        
        "PCA_Ref_Mag.X", "PCA_Ref_Mag.Y", "PCA_Ref_Mag.Z",
        "PCA_Ref_Vec0.X", "PCA_Ref_Vec0.Y", "PCA_Ref_Vec0.Z",
        "PCA_Ref_Vec1.X", "PCA_Ref_Vec1.Y", "PCA_Ref_Vec1.Z",
        "PCA_Ref_Vec2.X", "PCA_Ref_Vec2.Y", "PCA_Ref_Vec2.Z" // 58
    });
    for (int i = 0; i < rows.size(); i++) {
        file << rows[i];
        if (i < rows.size()-1) file << ",";
    }
    file << endl;
    
    for (size_t i = 0; i < cloud->size(); i++) {
        if (i % 10000 == 0) cout << "Write " << i << " / " << cloud->size() << endl;
        file << cloud->ids[i] << ",";
        
        file << cloud->points[i].x + cloud->offset.x << ",";
        file << cloud->points[i].y + cloud->offset.y << ",";
        file << cloud->points[i].z + cloud->offset.z << ",";
        file << cloud->reflectance[i] << ",";
        if (withClasses) {
            file << cloud->classes[i] << ",";
        }
//        file << fixed << setprecision(4);
        
        const auto &f = cloudFeatures[i];
        file << f.groundHeight << ",";
        file << f.nearestPointDist[0] << ",";
        file << f.nearestPointDist[1] << ",";
        file << f.nearestPointDist[2] << ",";
        file << f.nearestCount << ",";
        file << f.nearestCountSameRef << ",";
        
        writePoint(file, f.planeNormal);
        file << f.planeCurvature << ","; // 16
        
        writePoint(file, f.meanOffsetSmall);
        writePoint(file, f.valAndDirSmall[0]);
        writePoint(file, f.valAndDirSmall[1]);
        writePoint(file, f.valAndDirSmall[2]);
        writePoint(file, f.valAndDirSmall[3]); // 31
        
        writePoint(file, f.meanOffset);
        writePoint(file, f.valAndDir[0]);
        writePoint(file, f.valAndDir[1]);
        writePoint(file, f.valAndDir[2]);
        writePoint(file, f.valAndDir[3]); // 46
        
        writePoint(file, f.valAndDirSameRef[0]);
        writePoint(file, f.valAndDirSameRef[1]);
        writePoint(file, f.valAndDirSameRef[2]);
        writePoint(file, f.valAndDirSameRef[3], false); // 58
        file << endl;
    }
    file.close();
}

void PangolinDSOViewer::saveFeatures2() {
    ofstream file(origCloudPath + "_result_2");
    file << fixed << setprecision(5);
    const int size = pcl::FPFHSignature33::descriptorSize();
    for (size_t i = 0; i < cloudFeatures2.size(); i++) {
        const auto &f = cloudFeatures2[i];
        if (i % 10000 == 0) cout << "Write " << i << " / " << cloud->size() << endl;
        for (int ii = 0; ii < size; ii++) {
            if (ii > 0) file << ",";
            file << f.histogram[ii];
        }
        file << endl;
    }
    file.close();
}


void PangolinDSOViewer::loadPredictions() {
    processedTypes = readClassesCSV(predictionsPath);
    cout << processedTypes.size() << " <> " << cloud->points.size() << endl;
    updatePredictions();
}

float PangolinDSOViewer::calculateAcuracy() {
    if (cloud->classes.size() != processedTypes.size()) {
        return 0;
    }
    
    const auto &realTypes = cloud->classes;
    long correctCount = 0;
    for (int i = 0; i < processedTypes.size(); i++) {
        if (realTypes[i] == processedTypes[i]) correctCount++;
    }
    auto confusion = calculateConfusion(realTypes, processedTypes);
    printConfusion(confusion);
    
    return correctCount / (double)processedTypes.size();
}

void PangolinDSOViewer::updatePredictions() {
    generateKeyframes(cloud, kfPointsProcessed, 1, &processedTypes);
    settings_displayProcessed = true;
    lastAcc = calculateAcuracy();
    cout << "Accuracy: " << fixed << setprecision(6) << lastAcc << endl;
    
    if (cloud->classes.size() == 0) {
        // GT нету, выходим из генерации ошибок
        return;
    }
    
    vector<Point3f> errorPoints;
    for (size_t i = 0; i < processedTypes.size(); i++) {
        if (processedTypes[i] == cloud->classes[i]) continue;
        errorPoints.push_back(cloud->points[i]);
    }
    if (errorPointsKf == nullptr) {
        errorPointsKf = new KeyFrameDisplay();
    }
    errorPointsKf->cloud = move(errorPoints);
    errorPointsKf->pushDataToBuffers();
}


void PangolinDSOViewer::rotateGenerateAndSaveFeatures() {
    auto origPoints = cloud->points;
    auto origPath = origCloudPath;
    for (int angle = 45; angle < 360; angle+=45) {
        cout << "Processing " << angle << "º" << endl;
        double c = cos(angle * M_PI / 180.0);
        double s = sin(angle * M_PI / 180.0);
        cv::Mat r = (Mat_<double>(3,3) << c, -s, 0, s, c, 0, 0, 0, 1);
        cv::transform(origPoints, cloud->points, r);
        
        cout << "<< Features 1 >>" << endl;
        generateFeatures();
        cout << "<< Features 2 >>" << endl;
        generateFeatures2();
        cout << "<< Save features >>" << endl;
        origCloudPath = origPath + "_" + to_string(angle);
        saveFeatures();
        saveFeatures2();
    }
    cloud->points = move(origPoints);
    origCloudPath = origPath;
}

void PangolinDSOViewer::fixNoise() {
    pcl_algo::fixNoise(cloud, processedTypes);
    updatePredictions();
}
