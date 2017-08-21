//
//
//
//
//
//

#include "WrapperDarknet.h"


extern "C" void darknetPrepareNetwork(char * _model, char * _weights, network *_net) {

    cuda_set_device(0);

    *_net = parse_network_cfg(_model);
    load_weights(_net, _weights);


    set_batch_network(_net, 1);
}

extern "C" void darknetDetect(network *_net, image _im) {


}


WrapperDarknet::WrapperDarknet(std::string mModelFile, std::string mWeightsFile) {
    char *wStr1= new char[mModelFile.size() + 1];
    char *wStr2= new char[mWeightsFile.size() + 1];

    std::copy(mModelFile.begin(), mModelFile.end(), wStr1);
    std::copy(mWeightsFile.begin(), mWeightsFile.end(), wStr2);

    wStr1[mModelFile.size()] = '\0';
    wStr2[mWeightsFile.size()] = '\0';

    darknetPrepareNetwork(wStr1, wStr2, &mNet);

    delete[] wStr1;
    delete[] wStr2;

}

std::vector<std::vector<float> > WrapperDarknet::detect(const cv::Mat &img) {
    srand(2222222);

    IplImage* iplImg = new IplImage(img);

    unsigned char *data = (unsigned char *)iplImg->imageData;
    int h = iplImg->height;
    int w = iplImg->width;
    int c = iplImg->nChannels;
    int step = iplImg->widthStep;
    image im = make_image(w, h, c);
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }


    image sized = letterbox_image(im, mNet.w, mNet.h);
    layer l = mNet.layers[mNet.n-1];

    box *boxes = (box*) calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *) calloc(l.classes + 1, sizeof(float *));
    float **masks = 0;
    if (l.coords > 4){
        masks = (float**) calloc(l.w*l.h*l.n, sizeof(float*));
        for(int j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float*) calloc(l.coords-4, sizeof(float *));
    }

    float *X = sized.data;
    network_predict(mNet, X);
    float thresh = 0.24;
    float hier_thresh = 0.5;
    get_region_boxes(l, im.w, im.h, mNet.w, mNet.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
    float nms=0.3f;
    if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);



    std::vector<std::vector<float> > result;
    int num = l.w*l.h*l.n;
    for(int i = 0; i < num; ++i){
        std::vector<float> imgRes = {0};
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].

        int classi = max_index(probs[i], l.classes);
        float prob = probs[i][classi];
        if(prob > thresh){
            imgRes.push_back(classi);
            imgRes.push_back(prob);

            box b = boxes[i];
            float left  = (b.x-b.w/2.);
            float top   = (b.y-b.h/2.);
            float right = (b.x+b.w/2.);
            float bot   = (b.y+b.h/2.);

            imgRes.push_back(left);
            imgRes.push_back(top);
            imgRes.push_back(right);
            imgRes.push_back(bot);
            result.push_back(imgRes);
        }
    }

    //free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    return result;
}
