//
//
//
//
//
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif
extern "C" {
    #include "network.h"
    #include "darknet/src/image.h"
}

class WrapperDarknet{
public:
    WrapperDarknet(std::string mModelFile, std::string mWeightsFile);

    std::vector<std::vector<float> > detect(const cv::Mat& img);

private:
    list *mOptions;
    network mNet;
    box *mBoxes;
    float **mProbs;
    float **mMasks;
};

