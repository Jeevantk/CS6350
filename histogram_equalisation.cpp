#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "iostream"

using namespace std;
using namespace cv;

int * calculate_histogram(Mat src)
{
	int * histogram;
	histogram =new int[256];
	for(int i=0;i<256;i++)
	{
		histogram[i]=0;
	}
	

	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			histogram[src.at<uchar>(i,j)]++;
		}
	}

	return	histogram;
}

int * cdf(int * histogram)
{
	int * cdf;
	cdf=new int[256];
	int sum=0;
	for(int i=0;i<256;i++)
	{
		cdf[i]=0;
	}

	for(int i=0;i<256;i++)
	{
		cdf[i]=sum+histogram[i];
		sum=cdf[i];
	}

	return cdf;
}

Mat equalise_histogram(Mat src)
{
	int rows=src.rows;
	int cols=src.cols;

	int * hist=calculate_histogram(src);

	cout<<"Histogram Calculated \n";
	int * transform=cdf(hist);
	cout<<"rows: " <<rows << "\n cols: " << cols<<"\n";

	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			uchar &t=src.at<uchar>(i,j);
			t=255*transform[t]/(rows*cols);
		}
	}

	return src;
}

int main(int argc,char** argv)
{
	if(argc!=2){
		cout<<"Usage :"<<argv[0] << " <image location > \n" ;
		return -1;
	}

	Mat src=imread(argv[1],0);
	imshow("Original Image", src);
	waitKey(0);
	Mat equalised=equalise_histogram(src);

	
	imshow("Equalised Image",equalised);
	waitKey(0);
	

}
