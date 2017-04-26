//Author -------> Jeevan Thomas Koshy

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "iostream"

using namespace std;
using namespace cv;
using namespace xfeatures2d;


int main()
{

	//Reading the images
	Mat img1=imread("pan2.jpg");
	Mat img2=imread("pan1.jpg");


	Ptr<Feature2D> sift=SIFT::create();
	vector<KeyPoint> keypoints_1,keypoints_2;
	
	//detecting feature points
	sift->detect(img1,keypoints_1);
	sift->detect(img2,keypoints_2);

	Mat descriptor1,descriptor2;
	imshow("Image 1",img1);
	imshow("Image 2",img2);

	waitKey(0);


	//Computing Descriptors
	sift->compute(img1,keypoints_1,descriptor1);
	sift->compute(img2,keypoints_2,descriptor2);

	// Implementing Ratio Test. In order to implement ratio test in opencv use BFMatcher(Brute Force Matcher) with KNN=2(K nearest Neighbours)
	// After getting the best two matches we can compare their distances to apply my master ratio test.
	//Symmetric test is also inbuild in BFMatcher but will be of as default (turn that on by using crossCheck =True)

	BFMatcher matcher(NORM_L2); // Has to put the second parameter as True for implementing Symmetric Test . Currently its giving run time errors. Need to figure out a way to solve that
	vector<vector<DMatch> > matches;

	//Finding the Matches

	matcher.knnMatch(descriptor1,descriptor2, matches, 4);
	

	/*
	These matches are expected to have an accuracy of around 50%.
	I am planning to use 
	1. Ratio Test 
	2. Symmetric Test 
	3. RANSAC 
	Hoping that after these we will get correct correspondences in the images and 
	*/

	vector<DMatch> good_matches;
	for(int i=0;i<matches.size();i++)
	{
		const float ratio=0.9;
		if(matches[i][0].distance<ratio*matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}


	vector<Point2f> image1;
	vector<Point2f> image2;

	//Computing the image locations corresponding to the good matches found in order to input that to find the Homography Transformation

	for(int i=0; i< good_matches.size();i++)
	{
		image1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		image2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	//Applying RANSAC to remove all the outliers and to compute the HOMOGRAPHY transformation
	
	// instead of using this apply a RANSAC on features seperately and then apply Bundle Adjustment to obtain the camera parameters
	//
	Mat H=findHomography(image1,image2,CV_RANSAC);

	Mat result;

	//perspective transformation for the second image


	warpPerspective(img1,result,H,Size(img1.cols+img2.cols,img1.rows));


	Mat half(result,Rect(0,0,img2.cols,img2.rows));
	img2.copyTo(half);
	imwrite("myself.jpg",result);
	imshow("Result",result);
	
	waitKey(0);
	return 0;
}