//
//  confusion_matrix.hpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 06.10.2022.
//

#ifndef confusion_matrix_hpp
#define confusion_matrix_hpp

#include <stdio.h>
#include "../cloud/cloud.hpp"

std::vector<std::vector<int>> calculateConfusion(const std::vector<Type> &gt, const std::vector<Type> &pred);

void printConfusion(const std::vector<std::vector<int>> &confusion);

#endif /* confusion_matrix_hpp */
