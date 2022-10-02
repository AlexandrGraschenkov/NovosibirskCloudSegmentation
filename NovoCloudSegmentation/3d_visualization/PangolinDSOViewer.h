/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include <pangolin/pangolin.h>
#include <map>
#include <deque>
#include "SimpleMatrix.hpp"
#include "MatrixChnageHandler.hpp"
#include <thread>
#include <unordered_map>
#include <opencv2/highgui.hpp>
#include "cloud.hpp"

#include "TrajectoryDrawer.hpp"


class PangolinDSOViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h);
    virtual ~PangolinDSOViewer();
    
    void reset();
    
    void join();
    
    void close();
    void run();
    void prepare();
    
    void updateCloud(PointCloudRef cloud, float keepPercentSize);
    
private:
    PointCloudRef cloud;
    cv::Point3d kfGlobalOffset;
    std::unordered_map<Type, KeyFrameDisplay *> kfPoints;
    
    TrajectoryDrawer normalsTD;
    TrajectoryDrawer normalsEgoTD;
    MatrixChnageHandler *kbHandler;
    
    bool running;
    int w,h;
    std::mutex framesMtx;
    
//    std::thread runThread;

    void drawAll(float ptSize);
    void drawCoordSystem(float size);
    
    void customPredictSigns3dPose();
    void saveParametrsToFile();
    void loadParametrsToFile();
    void resetCameraPosition(pangolin::OpenGlRenderState &camera);
    
    // render settings
    bool settings_showKFCameras;
    bool settings_colorizeAverage;
    float settings_groundPointsSize;
    float settings_cracksPointsSize;
    bool settings_displaySegmentation;
    bool settings_displayAntennaOrientation;
    
    void runAntenaOrientationCalculation();
};

