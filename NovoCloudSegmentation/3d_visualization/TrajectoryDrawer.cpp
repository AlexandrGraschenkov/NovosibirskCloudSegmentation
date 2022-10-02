//
//  TrajectoryDrawer.cpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 04.04.2018.
//  Copyright © 2018 Alexander Graschenkov. All rights reserved.
//

#include "TrajectoryDrawer.hpp"

using namespace cv;

void TrajectoryDrawer::setPoints(const std::vector<Point3d> &vec) {
    vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, (GLuint)vec.size(), GL_DOUBLE, 3, GL_DYNAMIC_DRAW );
    vertexBuffer.Upload(&vec[0].x, sizeof(double)*3*vec.size(), 0);
}

void TrajectoryDrawer::setPoints(const std::vector<cv::Point3f> &vec) {
    vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, (GLuint)vec.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW );
    vertexBuffer.Upload(&vec[0].x, sizeof(float)*3*vec.size(), 0);
}

void TrajectoryDrawer::draw(float *color) {
    if (color != NULL) {
        glColor3f(color[0], color[1], color[2]);
    }
    
    pangolin::RenderVbo(vertexBuffer, oneDrawOneSkip ? GL_LINES : GL_LINE_STRIP);
//    vertexBuffer.Bind();
//    glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
//    glEnableClientState(GL_VERTEX_ARRAY);
//    int drawCount = vertexBuffer.num_elements;
//    if (oneDrawOneSkip) {
//        glDrawArrays(GL_LINES, 0, drawCount);
//    } else {
//        glDrawArrays(GL_LINE_STRIP, 0, drawCount);
//    }
//    glDisableClientState(GL_VERTEX_ARRAY);
//    vertexBuffer.Unbind();
}
void TrajectoryDrawer::drawTo(int idx, float *color) {
    if (color != NULL) {
        glColor3f(color[0], color[1], color[2]);
    }
    
    auto mode = oneDrawOneSkip ? GL_LINES : GL_LINE_STRIP;
    vertexBuffer.Bind();
    glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    glDrawArrays(mode, 0, MIN(vertexBuffer.num_elements, idx));
    
    glDisableClientState(GL_VERTEX_ARRAY);
    vertexBuffer.Unbind();
}
