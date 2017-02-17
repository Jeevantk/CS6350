#include "opencv/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "iostream"

using namespace std;
using namespace cv;
using namespace xfeatures2d;


int main()
{

	//Reading the images
	Mat img1=imread("pan1.jpg");
	Mat img2=imread("pan2.jpg");


	Ptr<Feature2D> sift=SIFT::create();
	vector<KeyPoint> keypoints_1,keypoints_2;
	
	//detecting feature points
	sift->detect(img1,keypoints_1);
	sift->detect(img2,keypoints_2);

	Mat descriptor1,descriptor2;

	//Computing Descriptors
	sift->compute(img1,keypoints_1,descriptor1);
	sift->compute(img2,keypoints_2,descriptor2);

	BFMatcher matcher;
	vector<DMatch> matches;

	//Finding the Matches 
	matcher.match(descriptor1,descriptor2,matches);

	/*
	These matches are expected to have an accuracy of around 50%.
	I am planning to use 
	1. Ratio Test 
	2. Symmetric Test 
	3. RANSAC 
	Hoping that after these we will get correct correspondences in the images and 
	*/

}