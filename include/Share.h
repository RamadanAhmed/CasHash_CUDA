#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>

#ifdef __CUDACC__
#define CUDA_UNIVERSAL_QUALIFIER __host__ __device__
#else
#define CUDA_UNIVERSAL_QUALIFIER
#endif

const int kDimSiftData = 128; // the number of dimensions of SIFT feature
const int kDimHashData = 128; // the number of dimensions of Hash code
const int kBitInCompHash = 64; // the number of Hash code bits to be compressed; in this case, use a <uint64_t> variable to represent 64 bits
const int kDimCompHashData = kDimHashData / kBitInCompHash; // the number of dimensions of CompHash code
const int kMinMatchListLen = 16; // the minimal list length for outputing SIFT matching result between two images
const int kMaxCntPoint = 4000; // the maximal number of possible SIFT points; ensure this value is not exceeded in your application

const int kCntBucketBit = 8; // the number of bucket bits
const int kCntBucketGroup = 6; // the number of bucket groups
const int kCntBucketPerGroup = 1 << kCntBucketBit; // the number of buckets in each group
const int kMaxMemberPerGroup = 100;

const int kCntCandidateTopMin = 6; // the minimal number of top-ranked candidates
const int kCntCandidateTopMax = 10; // the maximal number of top-ranked candidates
const int kMaxCandidatePerDist = 100;

typedef float SiftData_t; // CUDA GPUs are optimized for float arithmetics, we use float instead of int
typedef float* SiftDataPtr;
typedef const float* SiftDataConstPtr;
typedef uint8_t HashData_t;
typedef uint8_t* HashDataPtr; // Hash code is represented with <uint8_t> type; only the lowest bit is used
typedef uint64_t CompHashData_t;
typedef uint64_t* CompHashDataPtr; // CompHash code is represented with <uint64_t> type
typedef unsigned int BucketEle_t;
typedef unsigned int* BucketElePtr; // index list of points in a specific bucket

typedef std::pair<unsigned int, unsigned int> MatchPair_t;
typedef std::shared_ptr<MatchPair_t> MatchPairPtr;

typedef std::vector<MatchPair_t> MatchPairList_t;
typedef std::shared_ptr<MatchPairList_t> MatchPairListPtr;

template <typename T>
struct Matrix {
    int width;
    int height;
    size_t pitch; // row size in bytes
    T* elements;

    CUDA_UNIVERSAL_QUALIFIER inline T& operator() (int i, int j) {
        return *(reinterpret_cast<T *>(reinterpret_cast<char *>(elements) + i * pitch) + j);
    } // no more ugly pointer calcs

    CUDA_UNIVERSAL_QUALIFIER inline const T& operator() (int i, int j) const {
         return *(reinterpret_cast<T *>(reinterpret_cast<char *>(elements) + i * pitch) + j);
    }

    Matrix(int H, int W) : height(H), width(W){
        pitch = sizeof(T) * width; // init pitch, will be adjusted later if use cudaMallocPitch
    }

    Matrix() : width(0), height(0), pitch(0), elements(NULL) {
    }
};

struct ImageHost {
    int cntPoint; // the number of SIFT points
    std::string keyFilePath;
    Matrix<SiftData_t> siftData; // [cntPoint x 128] Matrix, storing all sift vectors one-off
    Matrix<CompHashData_t> compHashData; // [cntPoint x 2 Matrix]
    Matrix<HashData_t> bucketIDList; // element -> buckets [cntPoint x kCntBucketGroup]
    Matrix<BucketEle_t> bucketList; // bucket -> elements [kCntBucketGroup*kCntBucketPerGroup x kMaxMemberPerGroup]
};

struct ImageDevice {
    int cntPoint;
    Matrix<SiftData_t> siftData;
    Matrix<CompHashData_t> compHashData; // [cntPoint x 2 Matrix]
    Matrix<HashData_t> bucketIDList; // element -> buckets [cntPoint x kCntBucketGroup]
    Matrix<BucketEle_t> bucketList; // bucket -> elements [kCntBucketGroup*kCntBucketPerGroup x kMaxMemberPerGroup]
    std::map<int, BucketElePtr> targetCandidates;
    ~ImageDevice() {
        freeSiftData();
        freeHashData();
        freeBucketData();
    }
    void freeSiftData() {
        if(siftData.elements != nullptr) {
            cudaFree(siftData.elements);
            siftData.elements = nullptr;
        }
    }
    void freeHashData() {
        if(compHashData.elements != nullptr) {
            cudaFree(compHashData.elements);
            compHashData.elements = nullptr;
        }

    }
    void freeBucketData() {
        if(bucketIDList.elements != nullptr) {
            cudaFree(bucketIDList.elements);
            cudaFree(bucketList.elements);
            bucketList.elements = nullptr;
            bucketIDList.elements = nullptr;
        }
    }
    bool isEmpty() {
        return siftData.elements == nullptr;
    }

    void downloadHashData(Matrix<SiftData_t> *siftData) {
        // SiftData_t *h_Array = new T[count];
        // cudaMemcpy(h_Array, d_Array, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    void downloadBucketData(Matrix<HashData_t> bucketIDList, Matrix<BucketEle_t> bucketList) {
        
    }
};



#define CUDA_CHECK_ERROR                                                         \
    do {                                                                         \
        const cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                                \
            const char *const err_str = cudaGetErrorString(err);                 \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1     \
                      << ": " << err_str << " (" << err << ")" << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while(0)


template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CATCH_ERROR(val) check ( (val), #val, __FILE__, __LINE__)

template <typename T>
inline void dumpDeviceArray(T const *d_Array, int count) {
    T *h_Array = new T[count];
    cudaMemcpy(h_Array, d_Array, count * sizeof(T), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR;
    std::cout << "Dumping device array:\n";
    for(int i = 0; i < count; i++) {
        std::cout << +h_Array[i] << ", ";
    }
    std::cout << "[ " << count << " element(s) ]\n";
    delete [] h_Array;
}

template <typename T>
inline void dumpHostArray(T const *h_Array, int count) {
    std::cout << "Dumping host array:\n";
    for(int i = 0; i < count; i++) {
        std::cout << +h_Array[i] << ", ";
    }
    std::cout << "[ " << count << " element(s) ]\n";
}

