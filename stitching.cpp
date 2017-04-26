#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
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

using namespace cv;
using namespace std;
using namespace cv::detail;



int main(int argc,char* argv[])
{

	float conf_thresh=1.f;
	if(argc<3)
	{
		cout<<"Need Atleast Two images to Stitch"<<endl;
		cout<<argc<<endl;
		return -1;
	}

	int num_images=argc-1;

	//vetor to store all the imput images
	vector<Mat> images(num_images);


	Mat img;

	//Declares a pointer for Surf Feature Detection
	Ptr<FeaturesFinder> finder;

	finder = makePtr<SurfFeaturesFinder>();

	//Vector to store all the image features of all the images
	vector<ImageFeatures> features(num_images);

	


	cout<<"Reading the images and Computing the image features in the image"<<endl;
	for(int i=0;i<num_images;i++)
	{
		img=imread(argv[i+1]);

		if (img.empty())
        {
            cout<<"Can't open image " << argv[i+1]<<endl;
            return -1;
        }

		images.push_back(img);

		//computing the features of image i
		(*finder)(img, features[i]);

		features[i].img_idx = i;
	}

	finder->collectGarbage();

	cout<<"finding Matches in the images"<<endl;

	//Vector to store all the match imformation
	vector<MatchesInfo> pairwise_matches;

	Ptr<FeaturesMatcher> matcher;

	//initialising the matcher
	matcher=makePtr<BestOf2NearestMatcher>();

	(*matcher)(features, pairwise_matches);

	//Take only the images which we are sure from the same panaroma

	vector<int> indices = leaveBiggestComponent(features, pairwise_matches,conf_thresh);

	vector<Mat> img_subset;

	for(int i=0;i<indices.size();i++)
	{
		img_subset.push_back(images[indices[i]]);
	}

	images=img_subset;

	//Checking if we have atleast two images from the same subset

	num_images=images.size();
	if(num_images<2)
	{
		cout<<"Needs more images"<<endl;
		return -1;
	}

	//cout<<num_images<<endl;

	Ptr<Estimator> estimator;

	estimator=makePtr<HomographyBasedEstimator>();

	vector <CameraParams> cameras;

	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (int i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;

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

	vector<double> focals;

	for(int i=0;i< cameras.size();i++)
	{
		focals.push_back(cameras[i].focal);
	}

	//for finding median focal length
	sort(focals.begin(), focals.end());

	float warped_image_scale;


	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	vector<Mat> rmats;

	for(int i=0;i<cameras.size();i++)
	{
		rmats.push_back(cameras[i].clone());
	}

	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;

	waveCorrect(rmats, wave_correct);

	for(int i=0;i<cameras.size();i++)
	{
		cameras[i].R=rmats[i];
	}
	
	vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

	for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

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

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {

        full_img = imread(img_names[img_idx]);

        compose_work_aspect = compose_scale / work_scale;

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


            Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            Rect roi = warper->warpRoi(sz, K, cameras[i].R);
            corners[i] = roi.tl();
            sizes[i] = roi.size();
        }


        img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

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
            else
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
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



	return 1;
}