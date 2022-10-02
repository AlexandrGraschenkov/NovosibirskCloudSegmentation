//
//  cloud_analyzer.hpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 02.10.2022.
//

#ifndef cloud_analyzer_hpp
#define cloud_analyzer_hpp

#include <stdio.h>
#include "cloud.hpp"

namespace pcl_algo {
void processCloud(PointCloudRef cloud, std::vector<Type> &outTypes);
} // namespace pcl_algo

#endif /* cloud_analyzer_hpp */
