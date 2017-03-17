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
	if(argc<3)
	{
		cout<<"Need Atleast Two images to Stitch"<<endl;
		cout<<argc<<endl;
		return -1;
	}

	int num_images=argc-1;
	vector<Mat> images(num_images);
	Mat img;
	for(int i=0;i<num_images;i++)
	{
		img=imread(argv[i+1]);
		images.push_back(img);
	}


	Ptr<FeaturesFinder> finder;
	finder = makePtr<SurfFeaturesFinder>();




	return 1;
}