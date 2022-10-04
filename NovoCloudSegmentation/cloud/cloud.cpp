//
//  cloud.cpp
//  NovoCloudSegmentation
//
//  Created by Alexander Graschenkov on 01.10.2022.
//

#include "cloud.hpp"
#include <fstream>
#include <sys/stat.h>   // stat
#include <stdbool.h>    // bool type
#include <iostream>

using namespace std;
using namespace cv;

bool file_exists(const char *filename) {
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

template<typename T>
void saveArr(const std::vector<T> &vec, const std::string &path) {
    FILE *write_ptr = fopen(path.c_str(),"wb");  // w for write, b for binary

    uchar *data = (uchar *)vec.data();
    size_t size = sizeof(T) * vec.size();
    fwrite(data,size,1,write_ptr); // write 10 bytes from our buffer
    fclose(write_ptr);
}

template<typename T>
bool loadArr(std::vector<T> &vec, const std::string &path) {
    if (!file_exists(path.c_str())) {
        return false;
    }
    
    FILE *fileptr;
    long filelen;

    fileptr = fopen(path.c_str(), "rb");  // Open the file in binary mode
    fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
    filelen = ftell(fileptr);             // Get the current byte offset in the file
    rewind(fileptr);                      // Jump back to the beginning of the file

    vec.resize(filelen / sizeof(T));
    fread(vec.data(), sizeof(T) * vec.size(), 1, fileptr); // Read in the entire file
    fclose(fileptr); // Close the file
    return true;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ',')) {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

void writeCache(const std::string &fullPath, const PointCloud& cloud) {
    string idsPath = fullPath + "_ids";
    string pointsPath = fullPath + "_points";
    string reflectancePath = fullPath + "_ref";
    string typePath = fullPath + "_type";
    
    saveArr(cloud.ids, idsPath);
    saveArr(cloud.points, pointsPath);
    saveArr(cloud.reflectance, reflectancePath);
    if (!cloud.classes.empty()) {
        saveArr(cloud.classes, typePath);
    }
}

bool readCache(const std::string &fullPath, PointCloud& cloud) {
    string idsPath = fullPath + "_ids";
    string pointsPath = fullPath + "_points";
    string reflectancePath = fullPath + "_ref";
    string typePath = fullPath + "_type";
    
    bool ok = loadArr(cloud.ids, idsPath) &&
        loadArr(cloud.points, pointsPath) &&
        loadArr(cloud.reflectance, reflectancePath);
    if (!ok) {
        cloud.ids.clear();
        cloud.reflectance.clear();
        cloud.points.clear();
        return false;
    }
    loadArr(cloud.classes, typePath);
    return true;
}

PointCloudRef readCSV(const std::string &fullPath, bool useCache) {
    ifstream file(fullPath);
    vector<string> rows = getNextLineAndSplitIntoTokens(file);
    bool withClass = rows.back() == "Class";
    PointCloud result;
    if (useCache && readCache(fullPath, result)
//        && false
        ) {
        cout << "Loaded from cache" << endl;
        
        auto values = getNextLineAndSplitIntoTokens(file);
        Point3d p(stod(values[1]), stod(values[2]), stod(values[3]));
        result.offset = p;
        return make_shared<PointCloud>(result);
    }
    
    // read csv
    bool setZero = true;
    while (true) {
        auto values = getNextLineAndSplitIntoTokens(file);
        if (values.size() != rows.size()) break;
        result.ids.push_back(stoi(values[0]));
        Point3d p(stod(values[1]), stod(values[2]), stod(values[3]));
        if (setZero) {
            setZero = false;
            result.offset = p;
        }
        p -= result.offset;
//        if (p.x - p.y < 70) continue; //
//        if ( - p.x - p.y > 10) continue;
//        if (p.x + p.y > 10) continue;
        result.points.push_back(p);
        result.reflectance.push_back(stof(values[4]));
        if (withClass) {
            result.classes.push_back((Type)stoi(values[5]));
        }
    }
    writeCache(fullPath, result);
    return make_shared<PointCloud>(result);
}

cv::Vec3f colorForType(Type t) {
    switch (t) {
        case TypeUnknown:
            return Vec3f(1, 1, 1);
            break;
        case TypeGround:
            return Vec3f(0.6, 0.5, 0.1);
            break;
        case TypeSupports:
            return Vec3f(0.6, 0.0, 0.8);
            break;
        case TypeGreenery:
            return Vec3f(0.1, 0.9, 0.1);
            break;
        case TypeRails:
            return Vec3f(0.4, 0.4, 1);
            break;
        case TypeContactNetwork:
            return Vec3f(0, 0, 0.9);
            break;
        case TypeNoise:
            return Vec3f(0.7, 0.1, 0.1);
            break;
            
        default:
            break;
    }
}
