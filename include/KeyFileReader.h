#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include <vector>

class KeyFileReader {
public:
    KeyFileReader();
    ~KeyFileReader();
    void UploadImage(ImageDevice &imgDev, const int index);
    void UploadImageAll(ImageDevice &imgDev, const int index);
    void DownloadImageAll(ImageDevice &imgDev, const int index);
    cudaEvent_t UploadImageAsync(ImageDevice &imgDev, const int index, cudaEvent_t sync = 0);
    void AddKeyFile(const char *path);
    void OpenKeyList(const char *path);
    void ZeroMeanProc();

    int cntImage;
    
    std::vector<ImageHost> h_imageList_;
private:
    SiftData_t siftAccumulator_[kDimSiftData];
    int cntTotalVector_;
    cudaStream_t keyFileReaderStream_;
};
