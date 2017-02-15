#include "opencv/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "iostream"

using namespace std;
using namespace cv;
using namespace xfeatures2d;


int main()
{
	Ptr<Feature2D> sift=SIFT::create();
	vector<KeyPoint> keypoints_1,keypoints_2;
}