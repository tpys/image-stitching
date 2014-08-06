Image Stitching
============
1.Introduction
---------------
It's a common project in computer vision course, you can find it in stanford, washington and brown. I borrow skeleton code
from washington cse576.

This project include three main parts:
* detect interest points
* match interst points
* RANSAC refine matches
* stitch two image together

What I did in this project:
* implement sift detector and sift descriptor(this is crazy part, lots detail)
* use SSD matching points
* RANSAC to refine mathes, iterate 500 times
* computer homography then stich two iamge together
* use dynamic programming to find optimum seam, blend along this seam.


Platform: Qt5 vs2013 c++11

2.Result
----------
 Interest Points(can't show scale and orientation) <br>
![](https://github.com/tpys/image-stitching/raw/master/interest1.png) <br>
 Match left <br>
![](https://github.com/tpys/image-stitching/raw/master/match2.png) <br>
 Match right <br>
![](https://github.com/tpys/image-stitching/raw/master/match1.png) <br>
 Stitched <br>
![](https://github.com/tpys/image-stitching/raw/master/stitched.png) <br>

3.Issue
--------
*It take a lot time to build dog pyramid, I known there is a fast way to instead one big gauss kernel with two seperate samll gauss kernel. But, unsucess, don't why.
*In the matching part, some method such as knn can be used to reduce time, I didn't do that either.
*Finally my sift detector doesn't work with big rotation.
