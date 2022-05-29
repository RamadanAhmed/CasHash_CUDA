#pragma once
#include <memory>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include "Share.h"
#include "KeyFileReader.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
struct FeatureCache
{
    FeatureCache(unsigned long long physical_memory_percent)
    {
        min_available_physical_memory_ = 100 - physical_memory_percent;
        const auto available_memory = getAvailablePercentageGPUMemory();
        if (physical_memory_percent > available_memory)
            min_available_physical_memory_ = 100 - available_memory;
    }
    unsigned long long getAvailablePercentageGPUMemory()
    {
        cudaSetDevice(0);

        size_t l_free = 0;
        size_t l_Total = 0;
        cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);

        return 100 * (l_free * 1.0 / l_Total);
    }

    std::shared_ptr<ImageDevice> get(int index)
    {
        std::unique_lock<std::mutex> lck{mutex_};
        auto elem = cache_.find(index);
        if (elem != cache_.end() && !elem->second->isEmpty()) {
            return cache_[index];
        }
        while (!loadFeature(index))
        {
            prune(true);
        }
        return cache_[index];
    }

    bool load(KeyFileReader const &file_reader)
    {
        file_reader_ = file_reader;
        // upload till memeory of gpu is percentage of gpu memory is full
        // for(int i = 0; i < file_reader_.cntImage; ++i) {
        //     if(!loadFeature(i)) {
        //         // means not all features is loaded
        //         return false;
        //     }
        // }
        return true;
    }

    /**
     * \brief remove item(s) from cache in case its not used frequently
     * \param bPruneOnlyOne determine if we will remove one item or more
     * \return number of item removed
     */
    std::size_t prune(const bool bPruneOnlyOne = true) const
    {
        std::size_t count = 0;
        for (auto it = begin(cache_); it != end(cache_);)
        {
            if (it->second.use_count() == 1 && !it->second->isEmpty())
            {
                it->second->freeSiftData();
                ++count;
                if (bPruneOnlyOne)
                {
                    break;
                }
            }
            else
            {
                ++it;
            }
        }
        return count;
    }
    /**
     * \brief load feature to cache
     * \param index index of feature in the map_id_string
     * \return true if loaded successfully, false otherwise
     */
    bool loadFeature(const int index)
    {
        const unsigned __int64 physMem = getAvailablePercentageGPUMemory();
        if (physMem > min_available_physical_memory_)
        {
            std::shared_ptr<ImageDevice> image = std::make_shared<ImageDevice>();
            std::cerr << "---------------------\nUploading image #" << index << " to GPU...\n";
            file_reader_.UploadImage(*image, index);
            cache_[index] = image;
            return true;
        }
        return false;
    }

private:
    KeyFileReader file_reader_;
    mutable std::mutex mutex_; // To deal with multithread concurrent access
    unsigned __int64 min_available_physical_memory_;
    mutable std::unordered_map<unsigned long long, std::shared_ptr<ImageDevice>> cache_;
};