#include "main.h"
#include <iostream>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <Windows.h>
#include <mmsystem.h>
#pragma comment(lib, "Winmm.lib")

cv::Rect detectFace(IplImage *img, IplImage *obj, CvHaarClassifierCascade *cascade_f, CvMemStorage *storage){
	/* detect faces */
	CvSeq *faces = cvHaarDetectObjects(
		obj, /* the source image */
		cascade_f, /* the face classifier */
		storage, /* memory buffer, created with cvMemStorage */
		1.2, 3, 0, /* special parameters, tune for your app*/
		cvSize(40, 40) /* minimum detection scale */
		);
	// img에서 cascade_f에 따라 얼굴영역을 검출한다.

	/* return if not found */
	if (faces->total == 0){ return cv::Rect(0,0,0,0);}
	// img에서 얼굴영역이 검출 되지 않을 경우
		
	/* get the first detected face */
	CvRect *face = (CvRect*)cvGetSeqElem(faces, 0);
		
	/* draw a red rectangle */
	cvRectangle(
		img,
		cvPoint(face->x, face->y -20),
		cvPoint(
		face->x + face->width,
		face->y + face->height +20
		),
		CV_RGB(255, 0, 0),
		1, 8, 0
		);
	
	/* return the Region of Interest */
	return cv::Rect(face->x,face->y-20,face->width,face->height+20);
}

cv::Rect detectEye(IplImage *img, IplImage *obj, CvHaarClassifierCascade *cascade_e,CvMemStorage *storage, cv::Rect face){

	cv::Rect ret;
	cvClearMemStorage(storage);
	
	cvSetImageROI(obj, face);	// 얼굴영역의 정보를 갖는 face에 따라 ROI 설정
	/* detect faces */
	CvSeq *eyes = cvHaarDetectObjects(
		obj, /* the source image */
		cascade_e, /* the face classifier */
		storage, /* memory buffer, created with cvMemStorage */
		1.1, 3, 0, /* special parameters, tune for your app*/
		cvSize(20, 20) /* minimum detection scale */
		);
	//	얼굴영역에서 눈을 검출
	
	cvResetImageROI(img);

	/* return if not found */
	if (eyes->total == 0){	
		//sndPlaySoundA("C:\\Users\\nlp\\Desktop\\warning.wav",SND_ASYNC|SND_NODEFAULT);
		//system("pause");
		return cv::Rect(0,0,0,0);
	}
	// ROI에서 눈이 검출되지 않을 경우

	/* get the first detected face */
	CvRect *eye = (CvRect*)cvGetSeqElem(eyes, 0);
	
	ret = cv::Rect(face.x+eye->x, face.y+eye->y+10, eye->width, eye->height);
	// 눈 영역의 ROI값을 설정한다.

	/* draw a red rectangle */
	cvRectangle(
		img,
		cvPoint(ret.x-10, ret.y-10),
		cvPoint(ret.x + ret.width+10, ret.y + ret.height + 10),
		CV_RGB(0, 255, 0),
		2, 8, 0
		);
	
	/* reset buffer for the next object detection */
	cvClearMemStorage(storage);

	return ret;
}

cv::Rect detectMouth(IplImage *img, IplImage *obj, CvHaarClassifierCascade *cascade_m,CvMemStorage *storage, cv::Rect face){

	cv::Rect ret;
	cvClearMemStorage(storage);
	

	cv::Rect new_face ;
	new_face.x = face.x;
	new_face.y = (int)face.y + face.height *(1.5/3);
	new_face.width = face.width;
	new_face.height = (int)face.height*(1.5/3);
	

	cvSetImageROI(obj, new_face);	// 얼굴영역의 정보를 갖는 face에 따라 ROI 설정
	/* detect faces */
	CvSeq *mouth = cvHaarDetectObjects(
		obj, /* the source image */
		cascade_m, /* the face classifier */
		storage, /* memory buffer, created with cvMemStorage */
		1.1, 3, 0, /* special parameters, tune for your app*/
		cvSize(40, 40) /* minimum detection scale */
		);
	//	얼굴영역에서 눈을 검출
	
	cvResetImageROI(img);

	/* return if not found */
	if (mouth->total == 0){	
		//sndPlaySoundA("C:\\Users\\nlp\\Desktop\\warning.wav",SND_ASYNC|SND_NODEFAULT);
		//system("pause");
		return cv::Rect(0,0,0,0);
	}
	// ROI에서 눈이 검출되지 않을 경우

	/* get the first detected face */
	CvRect *mm = (CvRect*)cvGetSeqElem(mouth, 0);
	
	ret = cv::Rect(new_face.x+mm->x, new_face.y+mm->y -10, mm->width, mm->height);

	cvRectangle(
		img,
		cvPoint(ret.x, ret.y),
		cvPoint(ret.x + ret.width, ret.y + ret.height),
		CV_RGB(0, 0, 255),
		2, 8, 0
		);
	
	/* reset buffer for the next object detection */
	cvClearMemStorage(storage);

	return ret;
}