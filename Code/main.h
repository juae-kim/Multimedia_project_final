#include <iostream>
#include <stdlib.h>
#include <list>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <windows.h>

using namespace std;

#define CHECK_TIME_START QueryPerformanceFrequency ((_LARGE_INTEGER*)&freq); QueryPerformanceCounter((_LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((_LARGE_INTEGER*)&end); a=(float)((float) (end - start)/freq)
#define NUM_OF_MANI 5

typedef struct{
	cv::Rect eye;
	cv::Point pupil;
}eye_pupil_inform;

extern int x_cor, y_cor;
extern IplImage *frame_;


/*
Rect Class? = Field값으로 x,y,width,height를 갖는 class로 roi의 범위를 나타내는데 쓰인다.
Point Class? = Field값으로 x,y을 갖는 class로 이미지 상의 좌료를 나타내는데 쓰인다.

/* function to detect face */
cv::Rect detectFace(IplImage *img, IplImage *obj, CvHaarClassifierCascade *cascade_f, CvMemStorage *storage);

/* fucntion to detect eye */
cv::Rect detectEye(IplImage *origin, IplImage *obj, CvHaarClassifierCascade *cascade_e,CvMemStorage *storage, cv::Rect coords);
cv::Rect detectMouth(IplImage *img, IplImage *obj, CvHaarClassifierCascade *cascade_m,CvMemStorage *storage, cv::Rect face);
/* function to detect corner */
float * detectCorner(IplImage * img, float *Pupil, bool left);

/* function to detect coordination of pupil center */
cv::Point getPupilCenter(cv::Mat &image, cv::Rect roi);

/* function to check whether roi is correctly set */
bool valid_Check(cv::Rect roi);

/* function to caculate mean coordinates */
eye_pupil_inform get_interpolated_coords(list<eye_pupil_inform> ob, bool left);