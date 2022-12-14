//
//  cloud_to_kf.hpp
//  CellTowerAnalyzer
//
//  Created by Alexander Graschenkov on 09.06.2022.
//

#pragma once

#include <stdio.h>
#include "cloud.hpp"
#include "../3d_visualization/KeyFrameDisplay.h"

void generateKeyframes(PointCloudRef cloud, std::unordered_map<Type, KeyFrameDisplay *> &kfMap, float keepPercentSize = 1.0, std::vector<Type> *overrideTypes = nullptr);
