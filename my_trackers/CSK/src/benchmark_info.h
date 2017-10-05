/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <io.h>
using namespace std;
using namespace cv;


int load_video_info(string base_path, string video_name, vector<Rect> &groundtruthRect, vector<String> &fileName);

void getFiles(string path, vector<string>& files, vector<string>& names);
