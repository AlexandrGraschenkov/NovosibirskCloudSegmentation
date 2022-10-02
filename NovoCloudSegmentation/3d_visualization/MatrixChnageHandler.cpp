//
//  MatrixChnageHandler.cpp
//  PointCloudVisualizer
//
//  Created by Alexander Graschenkov on 20.12.17.
//  Copyright © 2017 Alexander Graschenkov. All rights reserved.
//

#include "MatrixChnageHandler.hpp"

using namespace std;

MatrixChnageHandler::MatrixChnageHandler(pangolin::OpenGlRenderState& cam_state): pangolin::Handler3D(cam_state) {
}

void MatrixChnageHandler::Keyboard(pangolin::View&, unsigned char key, int x, int y, bool pressed) {
//    if (!pressed) { return; }
//
//    const double dPos = 0.01;
//    MyMatrix trans;
//    if      (key == 'q') { trans = MyMatrix::translation(dPos, 0, 0); changes = true; }
//    else if (key == 'a') { trans = MyMatrix::translation(-dPos, 0, 0); changes = true; }
//    else if (key == 'w') { trans = MyMatrix::translation(0, dPos, 0); changes = true; }
//    else if (key == 's') { trans = MyMatrix::translation(0, -dPos, 0); changes = true; }
//    else if (key == 'e') { trans = MyMatrix::translation(0, 0, dPos); changes = true; }
//    else if (key == 'd') { trans = MyMatrix::translation(0, 0, -dPos); changes = true; }
//
//    const double dRot = 0.005;
//    if      (key == 'r') { trans = MyMatrix::rotation(dRot, 0, 0); changes = true; }
//    else if (key == 'f') { trans = MyMatrix::rotation(-dRot, 0, 0); changes = true; }
//    else if (key == 't') { trans = MyMatrix::rotation(0, dRot, 0); changes = true; }
//    else if (key == 'g') { trans = MyMatrix::rotation(0, -dRot, 0); changes = true; }
//    else if (key == 'y') { trans = MyMatrix::rotation(0, 0, dRot); changes = true; }
//    else if (key == 'h') { trans = MyMatrix::rotation(0, 0, -dRot); changes = true; }
//
//    const double dScale = 0.01;
//    if      (key == 'z') { trans = MyMatrix::scale(dScale+1, dScale+1, dScale+1); changes = true; }
//    else if (key == 'x') { trans = MyMatrix::scale(-dScale+1, -dScale+1, -dScale+1); changes = true; }
//
//    if (changes) {
//        this->tranform = this->tranform.apply(trans);
//    }
}


void MatrixChnageHandler::saveToFile(std::string path) {
    std::ofstream file(path);
    auto mat = this->cam_state->GetModelViewMatrix();
    for (int i = 0; i < 16; i++) {
        file << mat.m[i] << " ";
    }
    file.close();
}

bool MatrixChnageHandler::loadToFile(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    bool success = true;
    pangolin::OpenGlMatrix mat = this->cam_state->GetModelViewMatrix();
    for (int i = 0; i < 16; i++) {
        if (file >> mat.m[i]) {
            
        } else {
            success = false;
        }
    }
    file.close();
    
    if (success) {
        this->cam_state->SetModelViewMatrix(mat);
        prevMat = mat;
    }
    return success;
}

bool MatrixChnageHandler::checkChanges() {
    bool changes = false;
    pangolin::OpenGlMatrix mat = this->cam_state->GetModelViewMatrix();
    for (int i = 0; i < 16; i++) {
        if (prevMat.m[i] != mat.m[i]) changes = true;
    }
    prevMat = mat;
    return changes;
}

//void MatrixChnageHandler::MouseMotion(pangolin::View &v, int x, int y, int button_state) {
//    pangolin::Handler3D::Handler::MouseMotion(v, x, y, button_state);
//
//    pangolin::OpenGlMatrix mat = this->cam_state->GetProjectionMatrix();
//    for (int i = 0; i < 16; i++) {
//        if (prevMat.m[i] != mat.m[i]) changes = true;
//    }
//    prevMat = mat;
//}
//void MatrixChnageHandler::Mouse(pangolin::View&, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) {
//
//}
//
//void MatrixChnageHandler::MouseMotion(pangolin::View&, int x, int y, int button_state) {
//
//}
//
//void MatrixChnageHandler::Special(pangolin::View&, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float p3, float p4, int button_state) {
//    
//}

