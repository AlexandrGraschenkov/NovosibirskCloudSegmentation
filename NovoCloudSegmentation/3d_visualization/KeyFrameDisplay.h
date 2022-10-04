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


#ifndef kf_hpp
#define kf_hpp

#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include "../utils/json.hpp"

#include <sstream>
#include <fstream>
#include "SimpleMatrix.hpp"


struct FrameTime {
    double time;
    int frameId;
};

void p_drawCamera(float sz);

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    KeyFrameDisplay(const nlohmann::json &js, cv::Point3d offset = {}, double timeStart = 0, const std::vector<FrameTime> &times = {});
    KeyFrameDisplay(MyMatrix cam, const std::vector<cv::Point3f> points);
    KeyFrameDisplay(std::ifstream &file, cv::Point3d offset = {});
    KeyFrameDisplay() {};
	~KeyFrameDisplay();

	// renders cam & pointcloud.
    void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor=1);
	void drawPC(float pointSize, float *color);
    static void drawCamS(const MyMatrix &mat, float lineWidth = 1, float* color = 0, float sizeFactor=1);

    bool isKeyframe;
    std::vector<cv::Point3f> cloud;
    std::vector<cv::Vec3f> colors;
    std::vector<float> relBS;
    
    int frameId;
    double frameTime = 0;
    MyMatrix camera;
    
    cv::Point3d getCamPosition();
    
    void setMaxDistance(double dist);
    void setMinRelBS(float relBS);
    
    std::vector<int> connectionIdxs;

    void updateColors();
    
private:
public:
    pangolin::GlBuffer vertexBuffer;
    pangolin::GlBuffer colorBuffer;
    
    double maxDistance;
    float minRelBS;
    int toIdx;
    
    void readCamera(nlohmann::json camJs);
    
    void filterDrawPoints();
    void pushDataToBuffers();
    void updatePointsBuffer(const std::vector<cv::Point3f> &vec);
    void updateColorBuffer(const std::vector<cv::Vec3f> &vec);
    
};


#endif // kf_hpp
