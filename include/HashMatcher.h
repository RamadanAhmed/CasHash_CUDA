#pragma once

#include <cuda_runtime.h>
#include "Share.h"
#include "FeatureCache.h"
#include <vector>

constexpr BucketEle_t INVALID_CANDIDATE = ~0;
constexpr int MAX_COMPHASH_DISTANCE = ~(1 << (sizeof(int) * 8 - 1));
constexpr float MAX_SIFT_DISTANCE = 1.0e38f;
constexpr int POSSIBLE_CANDIDATES = 8;
constexpr int HASH_MATCHER_BLOCK_SIZE = 32;
constexpr int HASH_MATCHER_ITEMS_PER_THREAD = 2;

class HashMatcher {
public:
    HashMatcher(FeatureCache * cache);
    ~HashMatcher();
    int NumberOfMatch(int queryImageIndex, int targetImageIndex);
    MatchPairListPtr MatchPairList(int queryImageIndex, int targetImageIndex);
    void AddImage(const ImageDevice &d_Image); /* return value: image index */
    cudaEvent_t AddImageAsync(const ImageDevice &d_Image, cudaEvent_t sync = NULL);
    void releaseCandidates(int queryImageIndex);
private:
    //std::vector<ImageDevice> d_imageList_;
    // non-owing ptr
    FeatureCache * cache_;
    std::map< std::pair< int, int >, MatchPairListPtr > matchDataBase_;
    cudaStream_t hashMatcherStream_;
    std::size_t currentImages = 0;

    cudaEvent_t GeneratePair(int queryImageIndex, int targetImageIndex);
};
