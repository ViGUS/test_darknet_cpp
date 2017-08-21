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
    image sized = letterbox_image(_im, _net->w, _net->h);
    layer l = _net->layers[_net->n-1];

    box *boxes = (box*) calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *) calloc(l.classes + 1, sizeof(float *));
    float **masks = 0;
    if (l.coords > 4){
        masks = (float**) calloc(l.w*l.h*l.n, sizeof(float*));
        for(int j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float*) calloc(l.coords-4, sizeof(float *));
    }

    float *X = sized.data;
    network_predict(*_net, X);
    int thresh = 0.24;
    int hier_thresh = 0.5;
    get_region_boxes(l, _im.w, _im.h, _net->w, _net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
    float nms=0.3f;
    if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);



    int num = l.w*l.h*l.n;
    for(int i = 0; i < num; ++i){
        //std::vector<float> imgRes = {0};
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].

        int classi = max_index(probs[i], l.classes);
        float prob = probs[i][classi];
        if(prob > thresh){
            //imgRes.push_back(classi);
            //imgRes.push_back(prob);

            box b = boxes[i];
            int left  = (b.x-b.w/2.)*_im.w;
            int top   = (b.y-b.h/2.)*_im.h;
            int right = (b.x+b.w/2.)*_im.w;
            int bot   = (b.y+b.h/2.)*_im.h;


            if(left < 0) left = 0;
            if(right > _im.w-1) right = _im.w-1;
            if(top < 0) top = 0;
            if(bot > _im.h-1) bot = _im.h-1;

            //imgRes.push_back(left);
            //imgRes.push_back(top);
            //imgRes.push_back(right);
            //imgRes.push_back(bot);
            //result.push_back(imgRes);
        }
    }

    free_image(_im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

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
    cv::Mat oriImage;
    img.copyTo(oriImage);
    oriImage.convertTo(oriImage, CV_32FC3, 1.0f/255.0f);

    image im = {oriImage.cols, oriImage.rows, 3, (float*)oriImage.data};

    std::vector<std::vector<float> > result;
    return result;
}
