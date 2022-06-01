#include "KeyFileReader.h"
#include "Share.h"

#include <fstream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>
#include <filesystem>

KeyFileReader::KeyFileReader()
{
    std::memset(siftAccumulator_, 0, sizeof(siftAccumulator_));
    keyFileReaderStream_ = 0;
}

KeyFileReader::~KeyFileReader()
{
    std::vector<ImageHost>::iterator it;
    for (it = h_imageList_.begin(); it != h_imageList_.end(); ++it)
    {
        delete[] it->siftData.elements;
    }
}

void KeyFileReader::AddKeyFile(const char *path)
{
    FILE *keyFile = fopen(path, "r");
    if (keyFile == NULL)
    {
        fprintf(stderr, "Key file %s does not exist!\n", path);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Reading SIFT vector from %s\n", path);
    int cntPoint, cntDim;
    fscanf(keyFile, "%d%d", &cntPoint, &cntDim);
    if (cntDim != kDimSiftData)
    {
        fprintf(stderr, "Unsupported SIFT vector dimension %d, should be %d!\n", cntDim, kDimSiftData);
        exit(EXIT_FAILURE);
    }

    ImageHost newImage;
    newImage.cntPoint = cntPoint;
    newImage.keyFilePath = path;

    size_t requiredSize = cntPoint * cntDim;
    newImage.siftData.elements = new SiftData_t[requiredSize];
    newImage.siftData.width = cntDim;
    newImage.siftData.height = cntPoint;
    newImage.siftData.pitch = cntDim * sizeof(SiftData_t);
    if (newImage.siftData.elements == NULL)
    {
        fprintf(stderr, "Can't allocate memory for host image!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < cntPoint; i++)
    {
        //fscanf(keyFile, "%*f%*f%*f%*f"); // ignoring sift headers
        SiftDataPtr rowVector = newImage.siftData.elements + kDimSiftData * i;
        for (int j = 0; j < kDimSiftData; j++)
        {
            fscanf(keyFile, "%f", &rowVector[j]);
            siftAccumulator_[j] = siftAccumulator_[j] + rowVector[j];
        }
        cntTotalVector_++;
    }
    fclose(keyFile);
    h_imageList_.push_back(newImage);
    cntImage = h_imageList_.size();
}

void KeyFileReader::LoadDescriptorFile(std::string const& file) {
    std::ifstream desc_file (file, std::ios::binary);
    if(!desc_file.good()) return;
    fprintf(stderr, "Reading SIFT vector from %s\n", file.c_str());
    // read descriptor file
    std::size_t features_count;
    desc_file.read((char*)&features_count, 1*sizeof(std::size_t));
    ImageHost newImage;
    newImage.cntPoint = cntTotalVector_ = features_count;
    newImage.keyFilePath = file;

    size_t requiredSize = features_count * kDimSiftData;
    newImage.siftData.elements = new SiftData_t[requiredSize];
    newImage.siftData.width = kDimSiftData;
    newImage.siftData.height = features_count;
    newImage.siftData.pitch = kDimSiftData * sizeof(SiftData_t);
    std::vector<unsigned char> sift_data(requiredSize);
    desc_file.read((char*)sift_data.data(), requiredSize * sizeof(unsigned char));
    for(int i = 0; i < features_count; ++i) {
        for(int j = 0; j < kDimSiftData; ++j) {
            newImage.siftData.elements[j + kDimSiftData * i] = sift_data[j + kDimSiftData * i] * 1.0;
            siftAccumulator_[j] += newImage.siftData.elements[j + kDimSiftData * i];
        }
    }
    h_imageList_.push_back(newImage);
    cntImage = h_imageList_.size();
}

void KeyFileReader::LoadFeatures(std::string const& directory) {
    // iterate folder
    // for each desc file
    // read size_t as count
    // read 128 * count as sift_element
    for (const auto& dir_entry : std::filesystem::directory_iterator(directory)){
        auto file_path = dir_entry.path();
        if(file_path.extension() == ".desc") {
            LoadDescriptorFile(file_path.string());
        }
    }
}

void KeyFileReader::OpenKeyList(const char *path)
{
    FILE *keyList = fopen(path, "r");
    char keyFilePath[256];
    if (keyList == NULL)
    {
        fprintf(stderr, "Keylist file %s does not exist!\n", path);
        exit(EXIT_FAILURE);
    }
    while (fscanf(keyList, "%s", keyFilePath) > 0)
    {
        AddKeyFile(keyFilePath);
    }
    fclose(keyList);
}

void KeyFileReader::ZeroMeanProc()
{
    SiftData_t mean[kDimSiftData];

    for (int i = 0; i < kDimSiftData; i++)
    {
        mean[i] = siftAccumulator_[i] / cntTotalVector_;
    }

    std::vector<ImageHost>::iterator it;

    for (it = h_imageList_.begin(); it != h_imageList_.end(); ++it)
    {
        for (int i = 0; i < it->cntPoint; i++)
        {
            SiftDataPtr rowVector = &it->siftData(i, 0);
            for (int j = 0; j < kDimSiftData; j++)
            {
                rowVector[j] -= mean[j];
            }
        }
    }
}

void KeyFileReader::UploadImage(ImageDevice &d_Image, const int index)
{
    d_Image.cntPoint = h_imageList_[index].cntPoint;
    d_Image.siftData.width = kDimSiftData;
    d_Image.siftData.height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&(d_Image.siftData.elements),
                    &(d_Image.siftData.pitch),
                    d_Image.siftData.width * sizeof(SiftData_t),
                    d_Image.siftData.height);

    cudaMemcpy2D(d_Image.siftData.elements,
                 d_Image.siftData.pitch,
                 h_imageList_[index].siftData.elements,
                 h_imageList_[index].siftData.pitch,
                 h_imageList_[index].siftData.width * sizeof(SiftData_t),
                 h_imageList_[index].siftData.height,
                 cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;
}

void KeyFileReader::UploadImageAll(ImageDevice &d_Image, const int index)
{
    d_Image.cntPoint = h_imageList_[index].cntPoint;
    d_Image.siftData.width = kDimSiftData;
    d_Image.siftData.height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&(d_Image.siftData.elements),
                    &(d_Image.siftData.pitch),
                    d_Image.siftData.width * sizeof(SiftData_t),
                    d_Image.siftData.height);

    cudaMemcpy2D(d_Image.siftData.elements,
                 d_Image.siftData.pitch,
                 h_imageList_[index].siftData.elements,
                 h_imageList_[index].siftData.pitch,
                 h_imageList_[index].siftData.width * sizeof(SiftData_t),
                 h_imageList_[index].siftData.height,
                 cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;

    if (h_imageList_[index].bucketIDList.elements != nullptr)
    {
        // upload bucketIDList
        d_Image.bucketIDList.width = kCntBucketGroup;
        d_Image.bucketIDList.height = d_Image.cntPoint;
        cudaMallocPitch(&(d_Image.bucketIDList.elements),
                        &(d_Image.bucketIDList.pitch),
                        d_Image.bucketIDList.width * sizeof(HashData_t),
                        d_Image.bucketIDList.height);
        cudaMemcpy2D(d_Image.bucketIDList.elements,
                     d_Image.bucketIDList.pitch,
                     h_imageList_[index].bucketIDList.elements,
                     h_imageList_[index].bucketIDList.pitch,
                     h_imageList_[index].bucketIDList.width * sizeof(HashData_t),
                     h_imageList_[index].bucketIDList.height,
                     cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR;
    }
    if (h_imageList_[index].bucketList.elements != nullptr)
    {
        // upload bucketList
        d_Image.bucketList.width = kMaxMemberPerGroup;
        d_Image.bucketList.height = kCntBucketGroup * kCntBucketPerGroup;
        cudaMallocPitch(&(d_Image.bucketList.elements),
                        &(d_Image.bucketList.pitch),
                        d_Image.bucketList.width * sizeof(BucketEle_t),
                        d_Image.bucketList.height);
        cudaMemcpy2D(d_Image.bucketList.elements,
                     d_Image.bucketList.pitch,
                     h_imageList_[index].bucketList.elements,
                     h_imageList_[index].bucketList.pitch,
                     h_imageList_[index].bucketList.width * sizeof(BucketEle_t),
                     h_imageList_[index].bucketList.height,
                     cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR;
    }
    if (h_imageList_[index].compHashData.elements != nullptr)
    {
        // upload compHashData
        d_Image.compHashData.width = 2;
        d_Image.compHashData.pitch = sizeof(CompHashData_t) * 2;
        d_Image.compHashData.height = d_Image.cntPoint;
        cudaMalloc(&(d_Image.compHashData.elements),
                   d_Image.compHashData.pitch * d_Image.compHashData.height);
        cudaMemcpy(d_Image.compHashData.elements, h_imageList_[index].compHashData.elements, d_Image.compHashData.pitch * d_Image.compHashData.height, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR;
    }
}

void KeyFileReader::DownloadImageAll(ImageDevice &d_Image, const int index)
{
    if (h_imageList_[index].bucketIDList.elements == nullptr)
    {
        // Download BucketList ID
        h_imageList_[index].bucketIDList.width = kCntBucketGroup;
        h_imageList_[index].bucketIDList.height = d_Image.cntPoint;
        h_imageList_[index].bucketIDList.pitch = kCntBucketGroup * sizeof(HashData_t);
        h_imageList_[index].bucketIDList.elements = new HashData_t[kCntBucketGroup * d_Image.cntPoint];
        cudaMemcpy2D(h_imageList_[index].bucketIDList.elements,
                     h_imageList_[index].bucketIDList.pitch,
                     d_Image.bucketIDList.elements,
                     d_Image.bucketIDList.pitch,
                     d_Image.bucketIDList.width * sizeof(HashData_t),
                     d_Image.bucketIDList.height,
                     cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR;
    }
    if (h_imageList_[index].bucketList.elements == nullptr)
    {
        // Download BucketList
        h_imageList_[index].bucketList.width = kMaxMemberPerGroup;
        h_imageList_[index].bucketList.height = kCntBucketGroup * kCntBucketPerGroup;
        h_imageList_[index].bucketList.pitch = kMaxMemberPerGroup * sizeof(BucketEle_t);
        h_imageList_[index].bucketList.elements = new BucketEle_t[kMaxMemberPerGroup * kCntBucketGroup * kCntBucketPerGroup];
        cudaMemcpy2D(h_imageList_[index].bucketList.elements,
                     h_imageList_[index].bucketList.pitch,
                     d_Image.bucketList.elements,
                     d_Image.bucketList.pitch,
                     d_Image.bucketList.width * sizeof(BucketEle_t),
                     d_Image.bucketList.height,
                     cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR;
    }
    if (h_imageList_[index].compHashData.elements == nullptr)
    {
        // Download CompHashData
        h_imageList_[index].compHashData.width = 2;
        h_imageList_[index].compHashData.pitch = sizeof(CompHashData_t) * 2;
        h_imageList_[index].compHashData.height = d_Image.cntPoint;
        h_imageList_[index].compHashData.elements = new CompHashData_t[2 * d_Image.cntPoint];
        cudaMemcpy(h_imageList_[index].compHashData.elements,
                   d_Image.compHashData.elements,
                   d_Image.compHashData.pitch * d_Image.compHashData.height,
                   cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR;
    }
}

cudaEvent_t KeyFileReader::UploadImageAsync(ImageDevice &d_Image, const int index, cudaEvent_t sync)
{
    if (keyFileReaderStream_ == 0)
    {
        cudaStreamCreate(&keyFileReaderStream_);
    }

    if (sync)
    {
        cudaStreamWaitEvent(keyFileReaderStream_, sync, 0);
    }

    d_Image.cntPoint = h_imageList_[index].cntPoint;
    d_Image.siftData.width = kDimSiftData;
    d_Image.siftData.height = h_imageList_[index].cntPoint;

    cudaMallocPitch(&(d_Image.siftData.elements),
                    &(d_Image.siftData.pitch),
                    d_Image.siftData.width * sizeof(SiftData_t),
                    d_Image.siftData.height);

    cudaMemcpy2DAsync(d_Image.siftData.elements,
                      d_Image.siftData.pitch,
                      h_imageList_[index].siftData.elements,
                      h_imageList_[index].siftData.pitch,
                      h_imageList_[index].siftData.width * sizeof(SiftData_t),
                      h_imageList_[index].siftData.height,
                      cudaMemcpyHostToDevice,
                      keyFileReaderStream_);

    cudaEvent_t finish;
    cudaEventCreate(&finish);
    cudaEventRecord(finish, keyFileReaderStream_);

    CUDA_CHECK_ERROR;

    return finish;
}
