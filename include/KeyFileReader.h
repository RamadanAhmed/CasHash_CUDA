#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include <vector>

class KeyFileReader {
public:
    KeyFileReader();
    ~KeyFileReader();
    void UploadImage(ImageDataDevice &imgDev, const int index);
    void AddKeyFile(const char *path);
    void OpenKeyList(const char *path);
    void ZeroMeanProc();

    int cntImage;
    
private:
    std::vector<ImageDataHost> h_imageList_;
    SiftData_t siftAccumulator_[kDimSiftData];
    int cntTotalVector_;
};
