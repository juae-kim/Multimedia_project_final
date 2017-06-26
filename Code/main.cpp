#include "main.h"
#include <iostream>
#include <stdlib.h>
#include <opencv/cv.h>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
LPCSTR siren ="C:\\Users\\jju75\\Desktop\\warning.wav";
int histSize = 16;
float range[] = {0,255};
const float *ranges[] = {range};
int x_cor=0, y_cor=0;
int ex_x=-1, ex_y=-1;
clock_t start_t,current_t, program_start, program_end;
clock_t open_mouth_t, close_mouth_t;
clock_t for_blink_start_t, for_blink_t;
double duration = 0.0;
IplImage *frame_;
int blink_time_per_m;
bool blink_curr = false, blink_prev=false;
int BLINK_TIMES = 48;
double CLOSE_TIME  = 0.69;
int m_flag = 0;

bool start_flag  = false;
int getRadiusOfPupil(cv::Mat &image, cv::Mat &gray, cv::Point &center, int min, int max){
    cv::Mat temp;
    cv::Rect pupilRect;
    int resultRadius = min;
    int minMeanVal = 255;

	// To left;
    for(int radius = 1; radius <= max; radius++){
        pupilRect.x = center.x - radius/2;
        pupilRect.y = center.y - radius/2;
        pupilRect.width = radius;
        pupilRect.height = radius;

        if(pupilRect.x < 0) pupilRect.x = 0;
        else if(pupilRect.x + pupilRect.width > image.cols)
            pupilRect.x = image.cols - pupilRect.width;
        if(pupilRect.y < 0) pupilRect.y = 0;
        else if(pupilRect.y + pupilRect.height > image.rows)
            pupilRect.y = image.rows - pupilRect.height;
        
        int meanVal = cv::mean(gray(pupilRect))[0];
        if(minMeanVal >= meanVal){
            minMeanVal = meanVal;
            resultRadius = radius;
        }
    }

	// To right;
    
    return resultRadius;
}


int main(int argc, char *argv[]){
	
	list <eye_pupil_inform> L_tracer, R_tracer;
	const char *classifer = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
	const char *left_eye_classifier = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
	const char *right_eye_classifier = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
	const char *mouth_classifier = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml";
	int prev_sum_hist = 0;
    CvHaarClassifierCascade* cascade = 0;
	CvHaarClassifierCascade* left_eye_cascade = 0;
	CvHaarClassifierCascade* right_eye_cascade = 0;
	CvHaarClassifierCascade* mouth_cascade = 0;


	cv::Rect face, L_face_ROI, R_face_ROI, L_eye_ROI, R_eye_ROI, Mouth_ROI, face_ROI;
	cv::Mat converted;
	eye_pupil_inform L_tmp, R_tmp;

	IplImage *good;
	__int64 start, freq, end;
	float	resultTime = 0;	

    cascade = (CvHaarClassifierCascade*) cvLoad(classifer, 0, 0, 0 );
	left_eye_cascade = (CvHaarClassifierCascade*) cvLoad(left_eye_classifier, 0, 0, 0 );
	right_eye_cascade = (CvHaarClassifierCascade*) cvLoad(right_eye_classifier, 0, 0, 0 );
	mouth_cascade = (CvHaarClassifierCascade*) cvLoad(mouth_classifier, 0, 0, 0 );

	if(!cascade){
        std::cerr<<"error: cascade error!!"<<std::endl;
        return -1;
    }

	CvMemStorage* storage = 0;
    storage = cvCreateMemStorage(0);
    if(!storage){
        std::cerr<<"error: storage error!!"<<std::endl;
        return -2;
    }

    IplImage *frame=0, *_frame = 0;
    CvCapture *capture = 0;

    capture = cvCaptureFromCAM(0);

    if(!capture){
        std::cerr<<"error: Cannot open init webcam!"<<std::endl;
        return -3;
    }
	//CLOSE_TIME += 0.40;
    cvNamedWindow("haar example (exit = esc)",CV_WINDOW_AUTOSIZE);

	for_blink_start_t = clock();

	while(true){
		CHECK_TIME_START;
		frame = cvQueryFrame(capture);

		cvFlip(frame, frame, 1);
		frame_ = frame;

        if(!frame || cvWaitKey(5)==27) { break; }
		good  = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
		cvCvtColor(frame, good, CV_BGR2GRAY);

		face = detectFace(frame, good, cascade, storage);
	
		if(valid_Check(face)/* && frame->height/2<=face.height && frame->height*7/10>=face.height*/){				
			face_ROI = cv::Rect(face.x,face.y+face.height/2,face.width,face.height/2); //face detection
			L_face_ROI = cv::Rect(face.x,face.y+face.height/4,face.width/2,face.height/3); // left eye detection
			R_face_ROI = cv::Rect(face.x+face.width/2,face.y+face.height/4,face.width/2,face.height/3); //right eye detection
			
			L_eye_ROI = detectEye(frame, good, left_eye_cascade, storage, L_face_ROI);
			R_eye_ROI = detectEye(frame, good, right_eye_cascade, storage, R_face_ROI);

		current_t = clock();
			if((double)(current_t -for_blink_start_t)/CLOCKS_PER_SEC >60.0){ // checking the blink number during 60seconds
				if(blink_time_per_m > BLINK_TIMES ){
					sndPlaySoundA("./warning.wav",SND_ASYNC|SND_NOSTOP);//|SND_NODEFAULT);
					printf("Warning!!!! - Too many blink!!\n");
				}
				for_blink_start_t = clock();
				blink_time_per_m = 0;
			}
			
			if((double)(current_t - start_t)/CLOCKS_PER_SEC > CLOSE_TIME){ // checking eyes close time
				if(start_flag == true){
					sndPlaySoundA("./warning.wav",SND_ASYNC|SND_NOSTOP);//|SND_NODEFAULT);
					printf("Warning!!!! - Open your eyes!!\n");
				}
				start_flag = true;
			
			}

			if(L_face_ROI!= cv::Rect(0,0,0,0) &&
					(!(L_eye_ROI == cv::Rect(0,0,0,0) && R_eye_ROI == cv::Rect(0,0,0,0)))){
				start_t = clock();	
				blink_curr = false;
			}
			else{
				blink_curr = true;
			}
			if(blink_curr != blink_prev) {
				blink_time_per_m ++;
			}
			blink_prev = blink_curr;

			
			//mouth detection
		
			Mouth_ROI = detectMouth(frame, good, mouth_cascade, storage,face_ROI);
			IplImage *frame_m = cvCreateImage(cvGetSize(frame),8,3);
			cvCopy(frame, frame_m);
			cvSetImageROI(frame_m,Mouth_ROI);
			
			IplImage *mouth_gray  = cvCreateImage(cvGetSize(frame_m), IPL_DEPTH_8U, 1);
			cvCvtColor(frame_m, mouth_gray, CV_BGR2GRAY);
		
			
			Mat imgOriginal;
			imgOriginal = cvarrToMat(mouth_gray);
			
			MatND hist;
			calcHist( &imgOriginal, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
			
			int sum_hist = 0;
			for(int i=0; i<6 ; i++){
				sum_hist += (int)hist.at<float>(i);
			}
			if((sum_hist!=0 && prev_sum_hist!=0) && m_flag == 0 && prev_sum_hist !=0 && sum_hist > prev_sum_hist + 50){ // checking mouth open
				open_mouth_t = clock();	
				m_flag = 1;
			}
			if((sum_hist!=0 && prev_sum_hist!=0) && m_flag ==1 && prev_sum_hist !=0 && sum_hist < prev_sum_hist - 50){ //checking mouth close
				if(m_flag!=2) close_mouth_t = clock();	
				m_flag =2;
			}

			if(m_flag == 2 && (double)(close_mouth_t - open_mouth_t)/CLOCKS_PER_SEC > 5.6){ // checking yawn time
				sndPlaySoundA("./warning.wav", SND_ASYNC|SND_NOSTOP);
				printf("Warning!!!! - Yawn!!\n");
				m_flag = 0;
			}
			if(m_flag == 2 && (double)(close_mouth_t - open_mouth_t)/CLOCKS_PER_SEC <= 5.6){
				m_flag = 0;
			}
			prev_sum_hist = sum_hist;
			
			
			
		//	pupill detection
			
			if(valid_Check(L_eye_ROI) && valid_Check(R_eye_ROI)){
				converted = cv::Mat(frame);

				L_tmp.pupil = getPupilCenter(converted, L_eye_ROI);
				R_tmp.pupil = getPupilCenter(converted, R_eye_ROI);

				if(L_tmp.pupil.x == 0 && L_tmp.pupil.y == 0 
					|| R_tmp.pupil.x == 0 && R_tmp.pupil.y == 0 ){
						sndPlaySoundA("./warning.wav",SND_ASYNC|SND_NOSTOP);//|SND_NODEFAULT);
				}
			
				cvCircle(frame, L_tmp.pupil, 2, CV_RGB(255,0,0), 3, 8);
				cvCircle(frame, R_tmp.pupil, 2, CV_RGB(255,0,0), 3, 8);	
			}
						
			
		}
		cvShowImage("haar example (exit = esc)",frame);	
		resultTime = 0;

	}

    //memory free
    cvReleaseCapture(&capture);
    cvReleaseMemStorage(&storage);
    cvReleaseHaarClassifierCascade(&cascade);
    cvDestroyWindow("haar example (exit = esc)");

	return 0;
}

bool valid_Check(cv::Rect roi){ // ROI valid check

	if(roi.width*roi.height)
		return true;
	return false;
}
