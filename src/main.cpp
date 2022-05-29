#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "KeyFileReader.h"
#include "HashConverter.h"
#include "HashMatcher.h"

int main(int argc, char **argv) {
    if(argc != 3) {
        fprintf(stderr, "Usage: %s <list.txt> outfile\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    FeatureCache feat_cache(20);
    KeyFileReader keyFileReader;
    keyFileReader.OpenKeyList(argv[1]);
    keyFileReader.ZeroMeanProc();
    feat_cache.load(keyFileReader);
    std::cerr << "Initializing CUDA device...\n";
    cudaSetDevice(0);

    HashConverter hashConverter;
    HashMatcher hashMatcher(&feat_cache);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FILE *outFile = fopen(argv[2], "w");

    cudaEventRecord(start);

    for(int imageIndex = 0; imageIndex < keyFileReader.cntImage; imageIndex++) {
        ImageDevice newImage;

        
        cudaEvent_t kfFinishEvent = keyFileReader.UploadImageAsync(newImage, imageIndex);

        std::cerr << "Calculating compressed Hash Values for image #" << imageIndex << "\n"; 
        cudaEvent_t hcFinishEvent = hashConverter.CalcHashValuesAsync(newImage, kfFinishEvent);

        std::cerr << "Matching image #" << imageIndex << " with previous images...\n";
        hashMatcher.AddImageAsync(newImage, hcFinishEvent);

        for(int imageIndex2 = 0; imageIndex2 < imageIndex; imageIndex2++) {
            MatchPairListPtr mpList = hashMatcher.MatchPairList(imageIndex, imageIndex2);
            int pairCount = hashMatcher.NumberOfMatch(imageIndex, imageIndex2);

            fprintf(outFile, "%d %d\n%d\n", imageIndex2, imageIndex, pairCount);

            for(MatchPairList_t::iterator it = mpList->begin(); it != mpList->end(); it++) {
                fprintf(outFile, "%d %d\n", it->second, it->first);
            }
        }

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, start, stop);
    std::cerr << "Time elapsed: " << timeElapsed << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    fclose(outFile);

    return 0;
}
