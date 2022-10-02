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



#include <stdio.h>
#include <iostream>
//#include "matrix_helper.hpp"
#include "TrajectoryDrawer.hpp"

//#include <GL/glx.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include <pangolin/pangolin.h>
#include "KeyFrameDisplay.h"
#include "SimpleMatrix.hpp"

using namespace std;
using namespace cv;




void KeyFrameDisplay::readCamera(nlohmann::json camJs) {
    double *data = camera.m.data();
    for (int i = 0; i < camJs.size(); i++) {
        int x = i % 4;
        int y = i / 4; // our matrix column based
        data[x*4+y] = camJs[i].get<double>();
    }
//    camera = camera.inverse();
}

KeyFrameDisplay::KeyFrameDisplay(MyMatrix cam, const vector<Point3f> points) {
    camera = cam;
    cloud = points;
    isKeyframe = points.size() > 0;
    
    pushDataToBuffers();
}

KeyFrameDisplay::KeyFrameDisplay(const nlohmann::json &js, cv::Point3d offset, double timeStart, const std::vector<FrameTime> &times) {
    readCamera(js["camera"]);
    *((Point3d *)&camera.data()[12]) -= offset;

    maxDistance = -1;
    minRelBS = -1;
    
    frameId = js["id"].get<int>();
    frameTime = js["time"].get<double>();
    if (timeStart != 0) {
        frameId = round((frameTime - timeStart) * 30);
    }
    
    if (times.size()) {
        double dt = abs(frameTime - times[0].time);
        for (int i = 1; i < times.size(); i++) {
            double curr = abs(frameTime - times[i].time);
            if (curr < dt) {
                frameId = times[i].frameId;
                dt = curr;
            } else {
//                cout << "Fixed frame id " << frameId << endl;
                break;
            }
        }
    }
    

    if (js["cloud"].size()) {
        isKeyframe = true;
        
        auto points = js["cloud"];
        for (int i = 2; i < points.size(); i+=3) {
            cv::Point3d p(points[i-2].get<float>(), points[i-1].get<float>(), points[i].get<float>());
            p = camera.apply(p);
            cloud.push_back(p);
        }
        auto colorsJs = js["colors"];
        for (int i = 2; i < colorsJs.size(); i+=3) {
            cv::Vec3d p(colorsJs[i-2].get<int>() / 255.0,
                        colorsJs[i-1].get<int>() / 255.0,
                        colorsJs[i].get<int>() / 255.0);
            swap(p[0],p[2]);
            colors.push_back(p);
        }
        if (js.count("obs")) {
            auto obsJs = js["obs"];
            for (int i = 0; i < obsJs.size(); i++) {
                relBS.push_back(obsJs[i].get<float>());
            }
        }
        
        
        if (js.count("connections")) {
            auto connections = js["connections"];
            if (connections.size() && connections[0].is_number()) {
                for (int i = 0; i < connections.size(); i++) {
                    connectionIdxs.push_back(connections[i].get<int>());
                }
            }
        }
    } else {
        isKeyframe = false;
    }
    pushDataToBuffers();
}

void KeyFrameDisplay::pushDataToBuffers() {
    updatePointsBuffer(cloud);
    updateColorBuffer(colors);
    toIdx = cloud.size();
    
    minRelBS = -1;
    maxDistance = -1;
}

KeyFrameDisplay::KeyFrameDisplay(std::ifstream &file, cv::Point3d offset) {
    int pointsCount;
    Eigen::Matrix<double, 7, 1> camVec;
    file >> frameTime >> pointsCount;
    for (int i = 0; i < 7; i++) {
        file >> camVec(i);
    }
    
    camera = MyMatrix::sim3_vec(camVec);
    *((Point3d *)&camera.data()[12]) -= offset;
//    camera = MyMatrix::scale(0.00001, 0.00001, 0.00001).apply(camera);
    cloud.resize(pointsCount);
    colors.resize(pointsCount);
    relBS.resize(pointsCount);
    for (int i = 0; i < pointsCount; i++) {
        file >> cloud[i].x >> cloud[i].y >> cloud[i].z;
        file >> colors[i][0] >> colors[i][1] >> colors[i][2] >> relBS[i];
        
        colors[i] /= 255.f;
        cloud[i] = camera.apply(cloud[i]);
    }

    isKeyframe = cloud.size();
    pushDataToBuffers();
}

void KeyFrameDisplay::updateColors() {
    updateColorBuffer(colors);
}

void KeyFrameDisplay::updatePointsBuffer(const std::vector<Point3f> &vec) {
    if (vertexBuffer.num_elements != vec.size()) {
        vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, (GLuint)vec.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    }
    if (vec.size()) {
        vertexBuffer.Upload(&vec[0], sizeof(float)*3*vec.size(), 0);
    }
}

void KeyFrameDisplay::updateColorBuffer(const std::vector<Vec3f> &vec) {
    if (colorBuffer.num_elements != vec.size()) {
        colorBuffer.Reinitialise(pangolin::GlArrayBuffer, (GLuint)vec.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    }
    if (vec.size()) {
        colorBuffer.Upload(&vec[0], sizeof(float)*3*vec.size(), 0);
    }
}

void KeyFrameDisplay::setMaxDistance(double newDist) {
    if (newDist == maxDistance) { return; }
    
    maxDistance = newDist;
    filterDrawPoints();
}

void KeyFrameDisplay::setMinRelBS(float newRelBS) {
    if (newRelBS == minRelBS) return;
    
    minRelBS = newRelBS;
    filterDrawPoints();
//    for (int )
}

void KeyFrameDisplay::filterDrawPoints() {
//    int idxI = 0;
//    Point3d camPos = getCamPosition();
//    for (int i = 0; i < cloud.size(); i++) {
//        double distVal = norm(cloud[i] - camPos);
//        bool ok = distVal < maxDistance;
//        if (relBS.size() && ok) {
//            ok = relBS[i] > minRelBS;
//        }
//        
//        if (ok) {
//            indexes[idxI] = i;
//            idxI++;
//        }
//    }
//    toIdx = idxI;
//    updateIndexBuffer(vector<uint>(indexes.begin(), indexes.begin() + idxI));
}


Point3d KeyFrameDisplay::getCamPosition() {
    return camera.getTranslations();
}



KeyFrameDisplay::~KeyFrameDisplay()
{
}

vector<Point3d> generateDefaultDrawCameraPoints() {
    float fx = 459.079529, cx = 639.926636, fy = 622.527405, cy = 455.00705;
    float width = 1280, height = 720;
    vector<Point3d> points;
    Point3d p0 = Point3d(0, 0, 0);
    Point3d p1 = Point3d((0-cx)/fx, (0-cy)/fy, 1);
    Point3d p2 = Point3d((0-cx)/fx, (height-1-cy)/fy, 1);
    Point3d p3 = Point3d((width-1-cx)/fx, (height-1-cy)/fy, 1);
    Point3d p4 = Point3d((width-1-cx)/fx, (0-cy)/fy, 1);
    
    points.push_back(p0); points.push_back(p1);
    points.push_back(p0); points.push_back(p2);
    points.push_back(p0); points.push_back(p3);
    points.push_back(p0); points.push_back(p4);
    
    points.push_back(p4);
    points.push_back(p3);
    
    points.push_back(p3);
    points.push_back(p2);
    
    points.push_back(p2);
    points.push_back(p1);
    
    points.push_back(p1);
    points.push_back(p4);
    
    return points;
}

void p_drawCamera(float sz) {
    static TrajectoryDrawer *drawer;
    static float transformMatrix[16];
    if (drawer == NULL) {
        drawer = new TrajectoryDrawer();
        
        drawer->setPoints(generateDefaultDrawCameraPoints());
        drawer->oneDrawOneSkip = true;
        for (int i = 0; i < 16; i++) {
            transformMatrix[i] = 0;
        }
        transformMatrix[15] = 1;
    }
    
    transformMatrix[0] = sz; // xy 0 0
    transformMatrix[5] = sz; // xy 1 1
    transformMatrix[10] = sz; // xy 2 2
    
    glMultMatrixf((GLfloat*)transformMatrix);
    
    drawer->draw();
}

void KeyFrameDisplay::drawCamS(const MyMatrix &mat, float lineWidth, float* color, float sizeFactor) {
    glPushMatrix();
    {
        glMultMatrixd((GLdouble*)mat.data());

        if(color == 0)
            glColor3f(1,1,0);
        else
            glColor3f(color[0],color[1],color[2]);

        
        glLineWidth(lineWidth);
        p_drawCamera(sizeFactor);
    }
    glPopMatrix();
}

void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
{
    KeyFrameDisplay::drawCamS(camera, lineWidth, color, sizeFactor);
}

void render(pangolin::GlBuffer& vbo, pangolin::GlBuffer& cbo, bool drawColor, int mode, int count, int from = 0) {
    
    if (cbo.num_elements && drawColor) {
        cbo.Bind();
        glColorPointer(cbo.count_per_element, cbo.datatype, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }
    
    vbo.Bind();
    glVertexPointer(vbo.count_per_element, vbo.datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    
//    glDrawElements(mode, count, GL_UNSIGNED_INT, &indexes[0]);
    glDrawArrays(mode, from, count);
    
    
    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
    if (cbo.num_elements && drawColor) {
        glDisableClientState(GL_COLOR_ARRAY);
        cbo.Unbind();
    }
}

void KeyFrameDisplay::drawPC(float pointSize, float *color)
{
    if (pointSize <= 0) return;
	glDisable(GL_LIGHTING);

	glPushMatrix();
    {
		glPointSize(pointSize*1);
        glEnable(GL_POINT_SMOOTH);
    
        if (color != NULL) {
//            glColor3f(1, 1, 1);
////            glColor3f(0.6, 0.2, 0.9);
//        } else {
            glColor3f(color[0], color[1], color[2]);
        }
        
        render(vertexBuffer, colorBuffer, color == NULL, GL_POINTS, toIdx, 0);
    }
	glPopMatrix();
}

