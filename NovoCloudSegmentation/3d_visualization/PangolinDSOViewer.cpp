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
    pangolin::Var<bool> settings_generateFeatures("ui.Generate features",false,false);
    pangolin::Var<bool> settings_saveFeatures("ui.Save features",false,false);
    
    pangolin::Var<bool> settings_showSegmentation("ui.Segmentation",false,true);
    pangolin::Var<bool> settings_showProcessed("ui.Processed",false,true);
    


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
        bool changes = false;
        
        if (pangolin::Pushed(settings_resetCamPosition)) {
            resetCameraPosition(Visualization3D_camera);
        }
        if (pangolin::Pushed(settings_processCloud)) {
            runProcessCloud();
            settings_showProcessed = settings_displayProcessed;
        }
        if (pangolin::Pushed(settings_generateFeatures)) {
            generateFeatures();
        }
        if (pangolin::Pushed(settings_saveFeatures)) {
            saveFeatures();
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
    generateKeyframes(cloud, kfPointsProcessed, 1, &newTypes);
    settings_displayProcessed = true;
}


void PangolinDSOViewer::generateFeatures() {
    pcl_algo::generateFeatures(cloud, cloudFeatures);
}

inline void writePoint(ofstream &file, const Point3f &p) {
    file << p.x << ",";
    file << p.y << ",";
    file << p.z << ",";
}

void PangolinDSOViewer::saveFeatures() {
    ofstream file(origCloudPath + "_result");
    file << fixed << setprecision(4);
    for (size_t i = 0; i < cloud->size(); i++) {
        if (i % 10000 == 0) cout << "Write " << i << " / " << cloud->size() << endl;
        file << cloud->ids[i] << ",";
        
        file << cloud->points[i].x + cloud->offset.x << ",";
        file << cloud->points[i].y + cloud->offset.y << ",";
        file << cloud->points[i].z + cloud->offset.z << ",";
        file << cloud->reflectance[i] << ",";
        file << cloud->classes[i] << ",";
//        file << fixed << setprecision(4);
        
        const auto &f = cloudFeatures[i];
        file << f.groundHeight << ",";
        file << f.nearestPointDist << ",";
        file << f.nearestCount << ",";
        file << f.nearestCountSameRef << ",";
        writePoint(file, f.meanOffset);
        writePoint(file, f.valAndDir[0]);
        writePoint(file, f.valAndDir[1]);
        writePoint(file, f.valAndDir[2]);
        writePoint(file, f.valAndDir[3]);
        writePoint(file, f.valAndDirSameRef[0]);
        writePoint(file, f.valAndDirSameRef[1]);
        writePoint(file, f.valAndDirSameRef[2]);
        writePoint(file, f.valAndDirSameRef[3]);
        file << endl;
    }
    file.close();
}
