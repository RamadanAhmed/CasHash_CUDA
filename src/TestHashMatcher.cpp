#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "HashMatcher.h"
#include "HashConverter.h"
#include "KeyFileReader.h"
#include "Share.h"
#include "FeatureCache.h"

#define DEBUG_HASH_MATCHER
int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    FeatureCache feat_cache(20);
    KeyFileReader kf;
    std::cerr << "reading keylist\n";
    kf.OpenKeyList(argv[1]);
    std::cerr << "preprocessing to zero-mean vectors\n";
    kf.ZeroMeanProc();

    feat_cache.load(kf);

    std::cerr << "filling hash matrix" << '\n';
    HashConverter hc;

    HashMatcher hm(&feat_cache);
    std::ofstream log ("log.txt");
    for (int i = 0; i < kf.cntImage; i++) {
        //kf.UploadImage(curImg, i);
        auto sp_image = feat_cache.get(i);
        std::cerr << "Converting hash values\n";
        hc.CalcHashValues(*sp_image);
        //dumpDeviceArray(&curImg.compHashData(0, 0), 2);
        //dumpDeviceArray(&curImg.bucketIDList(0, 0), 6);

        std::cerr << "Adding image to hashmatcher\n";
        hm.AddImage(*sp_image);

        for(int j = 0; j < i; j++) {
#ifdef DEBUG_HASH_MATCHER
            
            log << hm.NumberOfMatch(i, j) << " match(es) found between image " << i << " and " << j << "\n";

            MatchPairListPtr mpList = hm.MatchPairList(i, j);
            for(MatchPairList_t::iterator it = mpList->begin(); it != mpList->end(); it++) {
                log << "(" << it->first << ", " << it->second << ") ";
            }
            log << std::endl;
#endif
        }
        hm.releaseCandidates(i);

    }

    return 0;
}
