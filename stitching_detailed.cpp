#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;


// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;




int main(int argc, char* argv[])
{

    if(argc<2)
    {
        cout<<"Need more images : "<<endl;
        cout<<"Usage:"<<argv[0]<<" location/to/image1  location/to/image2 ..."<<endl;
        return 0;
    }

    for(int i=1;i<argc;i++)
    {
        img_names.push_back(argv[i]);
    }


    int num_images = argc-1;

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;


    Ptr<FeaturesFinder> finder;

    finder = makePtr<SurfFeaturesFinder>();

    Mat full_img, img;

    vector<ImageFeatures> features(num_images);

    vector<Mat> images(num_images);

    vector<Size> full_img_sizes(num_images);

    double seam_work_aspect = 1;


    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            cout<<"Can't open image " << img_names[i]<<endl;
            return -1;
        }

        work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
        is_work_scale_set = true;

        resize(full_img, img, Size(), work_scale, work_scale);


        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
        seam_work_aspect = seam_scale / work_scale;
        is_seam_scale_set = true;


        (*finder)(img, features[i]);

        features[i].img_idx = i;

        //cout<<"Features in image #" << i+1 << ": " << features[i].keypoints.size()<<endl;

        resize(full_img, img, Size(), seam_scale, seam_scale);

        images[i] = img.clone();
    }

    finder->collectGarbage();
    full_img.release();
    img.release();

    //LOGLN("Finding features");

    //LOG("Pairwise matching");

    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;


    matcher = makePtr<BestOf2NearestMatcher>();


    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    //LOGLN("Pairwise matching ");

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;


    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());

    if (num_images < 2)
    {
        cout<<"Need more images"<<endl;
        return -1;
    }

    Ptr<Estimator> estimator;

    estimator = makePtr<HomographyBasedEstimator>();


    vector<CameraParams> cameras;


    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return -1;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        //LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    Ptr<detail::BundleAdjusterBase> adjuster;

    adjuster = makePtr<detail::BundleAdjusterRay>();

    adjuster->setConfThresh(conf_thresh);


    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    refine_mask(0,0) = 1;
    refine_mask(0,1) = 1;
    refine_mask(0,2) = 1;
    refine_mask(1,1) = 1;
    refine_mask(1,2) = 1;


    adjuster->setRefinementMask(refine_mask);


    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return -1;
    }

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        //LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;


    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, wave_correct);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

    //LOGLN("Warping images (auxiliary)... ");

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
    warper_creator = makePtr<cv::SphericalWarper>();

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    //LOGLN("Warping images");

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

    Ptr<SeamFinder> seam_finder;

    seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);


    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    //LOGLN("Compositing...");


    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        //LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                //LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }

            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        // Blend the current image

        blender->feed(img_warped_s, mask_warped, corners[img_idx]);

    }

    Mat result, result_mask;
    blender->blend(result, result_mask);
    imwrite(result_name, result);
    // imshow("final",result);
    // waitKey(0);

    // // imshow("stitched",result);
    // // waitKey(0);

    //LOGLN("Finished, total time: " " sec");
    return 0;
}
