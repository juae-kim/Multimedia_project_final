#include "main.h"
#include <iostream>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>


cv::Point getPupilCenter(cv::Mat &image, cv::Rect roi){
    cv::Point pupilCenter;
    cv::Mat hsv, hue, sat, val;
	// RGB to HSV
    cv::cvtColor(image(roi).clone(), hsv, CV_RGB2HSV);
     
    // Hue hue channel
    if(!hue.data)	hue.release();
    hue.create( hsv.size(), hsv.depth() );
     
    // Create saturation channel
    if(!sat.data)	sat.release();
    sat.create( hsv.size(), hsv.depth() );
     
    // Create hue channel
    if(!val.data)	val.release();
    val.create( hsv.size(), hsv.depth() );
     
    // Extract channels
    cv::Mat out[] = {hue, sat, val};
    int from_to[] = { 0,0, 1,1, 2,2 };
    cv::mixChannels( &hsv, 1, out, 3, from_to, 3 );
     
    // Estimate center of pupil to max of sat
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    cv::minMaxLoc(sat, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    pupilCenter.x = (int)minLoc.x+roi.x;
    pupilCenter.y = (int)minLoc.y+roi.y;
     
    // Display saturation channel
    cv::Rect pupilRect;

	int a, b, _1, _2,i,min,max	;


		hue.release(); sat.release(); val.release();
	

	
	cv::Point center = pupilCenter;
	int radius = 20;
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_RGB2GRAY);
	cv::Rect currentCenterRect(center.x - radius/2, center.y - radius/2, radius, radius);
    cv::Rect shiftingRect(0,0,radius, radius);
    cv::Point resultCenter = center;
    int minMeanVal = 255;
    cv::Rect finalRect(0,0,0,0);
    
    // Shift y-axis
    for (int shiftY = -radius/2; shiftY < radius/2; shiftY++) {
        // Shift x-axis
        for (int shiftX = -radius/2; shiftX < radius/2; shiftX++) {
            shiftingRect.x = currentCenterRect.x + shiftX;
            shiftingRect.y = currentCenterRect.y + shiftY;
            if((shiftingRect.x < 0) || (shiftingRect.y < 0)
               || (shiftingRect.x+shiftingRect.width > gray.cols)
               || (shiftingRect.y+shiftingRect.height > gray.rows))
                continue;
            int meanVal = cv::mean(gray(shiftingRect))[0];
            if(minMeanVal > meanVal){
                minMeanVal = meanVal;
                finalRect = shiftingRect;
            }
        }
    }
    // Return center of final rect
    resultCenter.x = finalRect.x+radius/2;
    resultCenter.y = finalRect.y+radius/2;

	resultCenter.y -= 2;

    return resultCenter;

  
}
