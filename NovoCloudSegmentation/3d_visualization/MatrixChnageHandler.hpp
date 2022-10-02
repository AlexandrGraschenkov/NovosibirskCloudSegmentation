//
//  MatrixChnageHandler.hpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 20.12.17.
//  Copyright © 2017 Alexander Graschenkov. All rights reserved.
//

#ifndef MatrixChnageHandler_hpp
#define MatrixChnageHandler_hpp

#include <stdio.h>
#include <pangolin/pangolin.h>
#include "SimpleMatrix.hpp"

struct MatrixChnageHandler : pangolin::Handler3D
{
    MatrixChnageHandler(pangolin::OpenGlRenderState& cam_state);
    
    void Keyboard(pangolin::View&, unsigned char key, int x, int y, bool pressed);
//    void MouseMotion(pangolin::View &v, int x, int y, int button_state);
    
    void saveToFile(std::string path);
    bool loadToFile(std::string path);
    
    bool checkChanges();
private:
    pangolin::OpenGlMatrix prevMat;
};

#endif /* MatrixChnageHandler_hpp */
