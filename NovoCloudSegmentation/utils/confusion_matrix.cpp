//
//  confusion_matrix.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 06.10.2022.
//

#include "confusion_matrix.hpp"
#include <unordered_map>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

std::vector<std::vector<int>> calculateConfusion(const std::vector<Type> &gt, const std::vector<Type> &pred) {
    
    vector<int> temp;
    const unordered_map<Type, int> indexes = {
        {TypeUnknown, 0},
        {TypeGround, 1},
        {TypeSupports, 2},
        {TypeGreenery, 3},
        {TypeRails, 4},
        {TypeContactNetwork, 5},
        {TypeNoise, 6},
    };
    for (int i = 0; i < TYPE_COUNT+1; i++) temp.push_back(0);
    vector<vector<int>> result;
    for (int r = 0; r < TYPE_COUNT+1; r++) {
        result.push_back(temp);
    }
    
    const size_t size = min(gt.size(), pred.size());
    for (size_t i = 0; i < size; i++) {
        result[indexes.at(gt[i])]
            [indexes.at(pred[i])] += 1;
    }
    return result;
}

void printConfusion(const std::vector<std::vector<int>> &confusion) {
    cout << "Confusion matrix:" << endl;
    vector<string> types = {"Unkn", "Ground", "Supp", "Green", "Rails", "CN", "Noise"};
    cout << right << setw(8) << setfill(' ') << "gt/pred";
    long total = 0;
    for (const auto &vec : confusion) for (int v : vec) total += v;
    
    // col description
    for (auto t : types) {
        cout << right << setw(8) << setfill(' ') << t;
    }
    cout << "|" << right << setw(9) << setfill(' ') << "err_perc";
    cout << endl;
    vector<long> columnSum(confusion.size());
    for (int r = 0; r < confusion.size(); r++) {
        // row description
        cout << right << setw(8) << setfill(' ') << types[r];
        
        // table
        long sum = 0;
        for (int c = 0; c < confusion[r].size(); c++) {
            if (r != c) {
                sum += confusion[r][c];
                columnSum[c] += confusion[r][c];
            }
            cout << right << setw(8) << setfill(' ') << confusion[r][c];
        }
        
        // err percent col
        cout << "|" << right << setw(8) << setfill(' ') << fixed << setprecision(2) << (10000 * sum / (double)total) / 100.0 << "%";
        cout << endl;
    }
    for (int i = 0; i < confusion.size()+2; i++) {
        cout << right << setw(8) << setfill('-') << "-";
    }
    cout << endl;
    
    // err percent row
    cout << right << setw(8) << setfill(' ') << "err_perc";
    for (long sum : columnSum) {
        cout << "|" << right << setw(6) << setfill(' ') << fixed << setprecision(2) << (10000 * sum / (double)total) / 100.0 << "%";
    }
    cout << endl;
}
