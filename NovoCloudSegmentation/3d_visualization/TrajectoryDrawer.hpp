//
//  TrajectoryDrawer.hpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 04.04.2018.
//  Copyright © 2018 Alexander Graschenkov. All rights reserved.
//

#ifndef TrajectoryDrawer_hpp
#define TrajectoryDrawer_hpp

#include <stdio.h>
#include "KeyFrameDisplay.h"

class TrajectoryDrawer {
public:
    bool oneDrawOneSkip;
    void setPoints(const std::vector<cv::Point3d> &points);
    void setPoints(const std::vector<cv::Point3f> &points);
    void draw(float *color = NULL);
    void drawTo(int idx, float *color = NULL);
    int getPointsCount() { return vertexBuffer.num_elements; };
    
private:
    pangolin::GlBuffer vertexBuffer;
};

#endif /* TrajectoryDrawer_hpp */
