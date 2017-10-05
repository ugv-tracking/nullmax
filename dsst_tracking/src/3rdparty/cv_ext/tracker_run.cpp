/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
// License Agreement
// For Open Source Computer Vision Library
// (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// + Author: Klaus Haag
// * Refactor file: Split into files/classes
// * Replace command line arguments
// * Add a variety of additional features
*/

#include "tracker_run.hpp"

#include <iostream>
#include <ctype.h>

#include "init_box_selector.hpp"
#include "cf_tracker.hpp"

using namespace cv;
using namespace std;
using namespace TCLAP;
using namespace cf_tracking;

TrackerRun::TrackerRun(string windowTitle) :
_windowTitle(windowTitle),
_cmd(_windowTitle.c_str(), ' ', "0.1"),
_debug(0)
{
    _tracker = 0;
}

TrackerRun::~TrackerRun()
{
    if (_resultsFile.is_open())
        _resultsFile.close();

    if (_tracker)
    {
        delete _tracker;
        _tracker = 0;
    }
}

bool TrackerRun::start()
{
    while (true)
    {
        if (init() == false)
            return false;
        if (run() == false)
            return false;

        if (!_paras.repeat || _exit)
            break;

        _boundingBox = _paras.initBb;
        _isTrackerInitialzed = false;
    }

    return true;
}

bool TrackerRun::init()
{

    if (_paras.showOutput)
        namedWindow(_windowTitle.c_str());

    _isPaused = _paras.paused;
    _frameIdx = 0;
    return true;
}

bool TrackerRun::run()
{
    bool success = true;

    std::cout << "Switch pause with 'p'" << std::endl;
    std::cout << "Step frame with 'c'" << std::endl;
    std::cout << "Select new target with 'r'" << std::endl;
    std::cout << "Update current tracker model at new location  't'" << std::endl;
    std::cout << "Quit with 'ESC'" << std::endl;

    while (true)
    {
        success = update();

        if (!success)
            break;
    }

    _cap.release();

    return true;
}

bool TrackerRun::update()
{
    int64 tStart = 0;
    int64 tDuration = 0;


    //STEP1 import new image
    if (!_isPaused || _frameIdx == 0 || _isStep)
    {
        _cap >> _image;

        if (_image.empty())
            return false;

        ++_frameIdx;
    }

    //STEP2 if not initialized, do initialization, else update
    if (!_isTrackerInitialzed)
    {
        if (!_hasInitBox)
        {
            Rect box;

            if (!InitBoxSelector::selectBox(_image, box))
                return false;

            _boundingBox = Rect_<double>(static_cast<double>(box.x),
                static_cast<double>(box.y),
                static_cast<double>(box.width),
                static_cast<double>(box.height));

            _hasInitBox = true;
        }

        tStart = getTickCount();
        _targetOnFrame = _tracker->reinit(_image, _boundingBox);
        tDuration = getTickCount() - tStart;

        if (_targetOnFrame)
            _isTrackerInitialzed = true;
    }
    else if (_isTrackerInitialzed && (!_isPaused || _isStep))
    {
        _isStep = false;

        if (_updateAtPos)
        {
            Rect box;

            if (!InitBoxSelector::selectBox(_image, box))
                return false;

            _boundingBox = Rect_<double>(static_cast<double>(box.x),
                static_cast<double>(box.y),
                static_cast<double>(box.width),
                static_cast<double>(box.height));

            _updateAtPos = false;

            std::cout << "UpdateAt_: " << _boundingBox << std::endl;
            tStart = getTickCount();

            //STEP 3 according to the current image and bbox, update target.
            _targetOnFrame = _tracker->updateAt(_image, _boundingBox);
            tDuration = getTickCount() - tStart;

            if (!_targetOnFrame)
                std::cout << "Target not found!" << std::endl;
        }
        else
        {
            tStart = getTickCount();
            _targetOnFrame = _tracker->update(_image, _boundingBox);
            tDuration = getTickCount() - tStart;
        }
    }

    double fps = static_cast<double>(getTickFrequency() / tDuration);
    printResults(_boundingBox, _targetOnFrame, fps);

    if (_paras.showOutput)
    {
        Mat hudImage;
        _image.copyTo(hudImage);
        rectangle(hudImage, _boundingBox, Scalar(0, 0, 255), 2);
        Point_<double> center;
        center.x = _boundingBox.x + _boundingBox.width / 2;
        center.y = _boundingBox.y + _boundingBox.height / 2;
        circle(hudImage, center, 3, Scalar(0, 0, 255), 2);

        stringstream ss;
        ss << "FPS: " << fps;
        putText(hudImage, ss.str(), Point(20, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));

        ss.str("");
        ss.clear();
        ss << "#" << _frameIdx;
        putText(hudImage, ss.str(), Point(hudImage.cols - 60, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));

        if (_debug != 0)
            _debug->printOnImage(hudImage);

        if (!_targetOnFrame)
        {
            cv::Point_<double> tl = _boundingBox.tl();
            cv::Point_<double> br = _boundingBox.br();

            line(hudImage, tl, br, Scalar(0, 0, 255));
            line(hudImage, cv::Point_<double>(tl.x, br.y),
                cv::Point_<double>(br.x, tl.y), Scalar(0, 0, 255));
        }

        imshow(_windowTitle.c_str(), hudImage);

        if (!_paras.imgExportPath.empty())
        {
            stringstream ssi;
            ssi << setfill('0') << setw(5) << _frameIdx << ".png";
            std::string imgPath = _paras.imgExportPath + ssi.str();

            try
            {
                imwrite(imgPath, hudImage);
            }
            catch (runtime_error& runtimeError)
            {
                cerr << "Could not write output images: " << runtimeError.what() << endl;
            }
        }

        char c = (char)waitKey(10);

        if (c == 27)
        {
            _exit = true;
            return false;
        }

        switch (c)
        {
        case 'p':
            _isPaused = !_isPaused;
            break;
        case 'c':
            _isStep = true;
            break;
        case 'r':
            _hasInitBox = false;
            _isTrackerInitialzed = false;
            break;
        case 't':
            _updateAtPos = true;
            break;
        default:
            ;
        }
    }

    return true;
}

void TrackerRun::printResults(const cv::Rect_<double>& boundingBox, bool isConfident, double fps)
{
    if (_resultsFile.is_open())
    {
        if (boundingBox.width > 0 && boundingBox.height > 0 && isConfident)
        {
            _resultsFile << boundingBox.x << ","
                << boundingBox.y << ","
                << boundingBox.width << ","
                << boundingBox.height << ","
                << fps << std::endl;
        }
        else
        {
            _resultsFile << "NaN, NaN, NaN, NaN, " << fps << std::endl;
        }

        if (_debug != 0)
            _debug->printToFile();
    }
}

void TrackerRun::setTrackerDebug(cf_tracking::TrackerDebug* debug)
{
    _debug = debug;
}
