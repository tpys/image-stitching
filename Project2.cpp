#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

#include "Matrix.h"

#include <vector>
#include <array>
#include <map>
#include <tuple>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <random>
#include <ctime>

using namespace std;
#define PI  3.14159265358979323846

template<typename T, typename T2>
inline
T saturate_cast(T2 num)
{
	if (num < 0)
		return 0;
	if (num>255)
		return 255;
	return static_cast<uchar>(num);
}


void MainWindow::BlackWhiteImage(QImage *image)
{
	int r, c;
	QRgb pixel;

	for (r = 0; r < image->height(); r++)
	{
		for (c = 0; c < image->width(); c++)
		{
			pixel = image->pixel(c, r);
			double red = (double)qRed(pixel);
			double green = (double)qGreen(pixel);
			double blue = (double)qBlue(pixel);

			// Compute intensity from colors - these are common weights
			double intensity = 0.3*red + 0.6*green + 0.1*blue;

			image->setPixel(c, r, qRgb((int)intensity, (int)intensity, (int)intensity));
		}
	}
}


void MainWindow::ResamplingImage(QImage &image,double scale)
{
	QImage buffer;
	int w = image.width();
	int h = image.height();
	int r, c;

	buffer = image.copy();
	// Reduce the image size.
	int w2 = (w-2) / scale+1;
	int h2 = (h-2) / scale+1;
	image = QImage(w2, h2, QImage::Format_RGB32);

	// Copy every other pixel
	for (r = 0; r < h2; r++)
	for (c = 0; c < w2; c++)
	{
		//BilinearInterpolation(&buffer, scale * c, scale * r, rgb);

		QRgb pixel = buffer.pixel((int)(scale*c+0.5), (int)(scale*r+0.5));
		int rgb[3];

		rgb[0] = qRed(pixel);
		rgb[1] = qGreen(pixel);
		rgb[2] = qBlue(pixel);

		image.setPixel(c, r, qRgb(rgb[0], rgb[1], rgb[2]));
	}

}

std::vector<double> MainWindow::DownSampling(std::vector<double>& image, int& width, int& height, double scale)
{
	int w2 = width / scale;
	int h2 = height / scale;

	vector<double> result(w2*h2);

	for (int r = 0; r < h2; r++)
	for (int c = 0; c < w2; c++)
	{
		result[r*w2 + c] = image[(int)(scale*r)*width+(int)(scale*c)];
	}

	width = w2;
	height = h2;

	return result;
}


std::vector<double> MainWindow::UpSampling(std::vector<double>& image, int& width, int& height, double scale)
{
	int w2 = width*scale;
	int h2 = height*scale;

	vector<double> result(w2*h2);

	for (int r = 0; r < h2; r++)
	for (int c = 0; c < w2; c++)
	{
		result[r*w2 + c] = image[(int)(r/scale)*width + (int)(c/scale)];
	}

	/*
	A B C
	E F G
	H I J
	pixels A C H J are pixels from original image
	pixels B E G I F are interpolated pixels
	*/
	// interpolate pixels B and I  
	for (int r = 0; r < h2; r += 2)
	for (int c = 1; c < w2 - 1; c += 2)
		result[r*w2 + c] = 0.5*(image[r/2*width + c/2] + image[r/2*width + c/2 + 1]);
	// interpolate pixels E and G  
	for (int r = 1; r < h2 - 1; r += 2)
	for (int c = 0; c < w2; c += 2)
		result[r*w2 + c] = 0.5*(image[r/2*width + c/2] + image[(r/2+1)*width + c/2]);
	// interpolate pixel F  
	for (int r = 1; r < h2 - 1; r += 2)
	for (int c = 1; c < w2 - 1; c += 2)
		result[r*w2+c] = 0.25*(image[r/2*width+c/2]+image[(r/2+1)*width+c/2]+image[r/2*width+c/2+1]+image[(r/2+1)*width+c/2+1]);


	width = w2;
	height = h2;

	return result;
}


bool MainWindow::Bilinear(std::vector<double>& image,int width, int height, double x, double y, double& result)
{
	if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1)
		return false;

	int x_left = static_cast<int>(x);
	int x_right = x_left + 1;
	int y_left = static_cast<int>(y);
	int y_right = y_left + 1;

	double f00 = image[y_left*width + x_left];
	double f10 = image[y_left*width + x_right];
	double f01 = image[y_right*width + x_left];
	double f11 = image[y_right*width + x_right];

	double r0 = (x_right - x)*f00 + (x - x_left)*f10;
	double r1 = (x_right - x)*f01 + (x - x_left)*f11;

	result = (y_right - y)*r0 + (y - y_left)*r1;

	return true;
}


/*******************************************************************************
Draw detected Harris corners
    interestPts - interest points
    numInterestsPts - number of interest points
    imageDisplay - image used for drawing

    Draws a red cross on top of detected corners
*******************************************************************************/
void MainWindow::DrawInterestPoints(CIntPt *interestPts, int numInterestsPts, QImage &imageDisplay)
{
   int i;
   int r, c, rd, cd;
   int w = imageDisplay.width();
   int h = imageDisplay.height();

   for(i=0;i<numInterestsPts;i++)
   {
       c = (int) interestPts[i].m_X;
       r = (int) interestPts[i].m_Y;

       for(rd=-2;rd<=2;rd++)
           if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
               imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

       for(cd=-2;cd<=2;cd++)
           if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
               imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
   }
}


void MainWindow::DrawInterestPoints(std::vector<std::tuple<std::pair<double, double>, double, double, std::vector<double>>> & interest_points,
	QImage &imageDisplay)
{

	int w = imageDisplay.width();
	int h = imageDisplay.height();

	int num = interest_points.size();
	for (int i = 0; i < num; i++)
	{
		int c = round(get<0>(interest_points[i]).second);
		int r = round(get<0>(interest_points[i]).first);

		for (int rd = -2; rd <= 2; rd++)
		if (r + rd >= 0 && r + rd < h && c >= 0 && c < w)
			imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

		for (int cd = -2; cd <= 2; cd++)
		if (r >= 0 && r < h && c + cd >= 0 && c + cd < w)
			imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
	}

}


/*******************************************************************************
Compute interest point descriptors
    image - input image
    interestPts - array of interest points
    numInterestsPts - number of interest points

    If the descriptor cannot be computed, i.e. it's too close to the boundary of
    the image, its descriptor length will be set to 0.

    I've implemented a very simple 8 dimensional descriptor.  Feel free to
    improve upon this.
*******************************************************************************/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *interestPts, int numInterestsPts)
{
    int r, c, cd, rd, i, j;
    int w = image.width();
    int h = image.height();
   // double *buffer = new double [w*h];

	vector<double> buffer(w*h);
	
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Blur
    SeparableGaussianBlurImage(buffer, w, h, sigma);

    // Compute the desciptor from the difference between the point sampled at its center
    // and eight points sampled around it.
    for(i=0;i<numInterestsPts;i++)
    {
        int c = (int) interestPts[i].m_X;
        int r = (int) interestPts[i].m_Y;

        if(c >= rad && c < w - rad && r >= rad && r < h - rad)
        {
            double centerValue = buffer[(r)*w + c];
            int j = 0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                    if(rd != 0 || cd != 0)
                {
                    interestPts[i].m_Desc[j] = buffer[(r + rd*rad)*w + c + cd*rad] - centerValue;
                    j++;
                }

            interestPts[i].m_DescSize = DESC_SIZE;
        }
        else
        {
            interestPts[i].m_DescSize = 0;
        }
    }

  //  delete [] buffer;
}

/*******************************************************************************
Draw matches between images
    matches - matching points
    numMatches - number of matching points
    image1Display - image to draw matches
    image2Display - image to draw matches

    Draws a green line between matches
*******************************************************************************/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display)
{
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }

}


void MainWindow::DrawMatches(std::vector<std::pair<double,double>>& feature0, 
	std::vector<std::pair<double, double>>& feature1, 
	QImage &image1Display,
	QImage &image2Display)
{
	int i;
	// Show matches on image
	QPainter painter;
	painter.begin(&image1Display);
	QColor green(0, 250, 0);
	QColor red(250, 0, 0);

	int num_match = feature0.size();

	for (i = 0; i < num_match; i++)
	{
		painter.setPen(green);
		painter.drawLine((int)feature0[i].second, (int)feature0[i].first, (int)feature1[i].second, (int)feature1[i].first);
		painter.setPen(red);
		painter.drawEllipse((int)feature0[i].second - 1, (int)feature0[i].first - 1, 3, 3);
	}

	QPainter painter2;
	painter2.begin(&image2Display);
	painter2.setPen(green);

	for (i = 0; i < num_match; i++)
	{
		painter2.setPen(green);
		painter2.drawLine((int)feature0[i].second, (int)feature0[i].first, (int)feature1[i].second, (int)feature1[i].first);
		painter2.setPen(red);
		painter2.drawEllipse((int)feature1[i].second - 1, (int)feature1[i].first - 1, 3, 3);
	}

}


/*******************************************************************************
Given a set of matches computes the "best fitting" homography
    matches - matching points
    numMatches - number of matching points
    h - returned homography
    isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*******************************************************************************/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward)
{
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }


        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}

bool MainWindow::ComputeHomography(std::vector<std::pair<double, double>>& feature0, 
	std::vector<std::pair<double, double>>& feature1,
	double h[3][3],
	bool isForward)
{

	int error;
	int nEq = feature0.size() * 2;

	dmat M = newdmat(0, nEq, 0, 7, &error);
	dmat a = newdmat(0, 7, 0, 0, &error);
	dmat b = newdmat(0, nEq, 0, 0, &error);

	double x0, y0, x1, y1;

	for (int i = 0; i < nEq / 2; i++)
	{
		if (isForward == false)
		{
			x0 = feature0[i].second;
			y0 = feature0[i].first;
			x1 = feature1[i].second;
			y1 = feature1[i].first;
		}
		else
		{
			x0 = feature1[i].second;
			y0 = feature1[i].first;
			x1 = feature0[i].second;
			y1 = feature0[i].first;
		}


		//Eq 1 for corrpoint
		M.el[i * 2][0] = x1;
		M.el[i * 2][1] = y1;
		M.el[i * 2][2] = 1;
		M.el[i * 2][3] = 0;
		M.el[i * 2][4] = 0;
		M.el[i * 2][5] = 0;
		M.el[i * 2][6] = (x1*x0*-1);
		M.el[i * 2][7] = (y1*x0*-1);
		//M.el[i * 2][6] = -x1*x0;
		//M.el[i * 2][7] = -y1*x0;



		b.el[i * 2][0] = x0;
		//Eq 2 for corrpoint
		M.el[i * 2 + 1][0] = 0;
		M.el[i * 2 + 1][1] = 0;
		M.el[i * 2 + 1][2] = 0;
		M.el[i * 2 + 1][3] = x1;
		M.el[i * 2 + 1][4] = y1;
		M.el[i * 2 + 1][5] = 1;
		M.el[i * 2 + 1][6] = (x1*y0*-1);
		M.el[i * 2 + 1][7] = (y1*y0*-1);
		//M.el[i * 2 + 1][6] = -x1*y0;
		//M.el[i * 2 + 1][7] = -y1*y0;

		b.el[i * 2 + 1][0] = y0;

	}
	int ret = solve_system(M, a, b);
	if (ret != 0)
	{
		freemat(M);
		freemat(a);
		freemat(b);

		return false;
	}
	else
	{
		h[0][0] = a.el[0][0];
		h[0][1] = a.el[1][0];
		h[0][2] = a.el[2][0];

		h[1][0] = a.el[3][0];
		h[1][1] = a.el[4][0];
		h[1][2] = a.el[5][0];

		h[2][0] = a.el[6][0];
		h[2][1] = a.el[7][0];
		h[2][2] = 1;
	}

	freemat(M);
	freemat(a);
	freemat(b);

	return true;
}



/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/


/*******************************************************************************
Blur a single channel floating point image with a Gaussian.
    image - input and output image
    w - image width
    h - image height
    sigma - standard deviation of Gaussian

    This code should be very similar to the code you wrote for assignment 1.
*******************************************************************************/
std::vector<double> MainWindow::SeparableGaussianBlurImage(vector<double>& image, int w, int h, double sigma)
{
    // Add your code here

	if (sigma == 0) return vector<double>();

	int radius = 3*sigma;
	int size = 2 * radius + 1;
	sigma /= sqrt(2.0);


	std::vector<double> buffer((w + 2 * radius)*(h + 2 * radius));
	vector<double> result(w*h);

	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			buffer[(r + radius)*(w + 2 * radius) + c + radius] = image[r*w + c];
		}
	}

	double* kernel = new double[size];

	double Z = sqrt(2 * PI)*sigma;

	for (int x = -radius; x <= radius; ++x)
	{
		kernel[x + radius] = exp(-(pow(x, 2.0)) / (2 * pow(sigma, 2))) / Z;
	}

	double denom = 0.000001;


	for (int i = 0; i < size; ++i)
	{
		denom += kernel[i];
	}

	for (int i = 0; i < size; i++)
	{
		kernel[i] /= denom;
	}

	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double rgb[3];

			rgb[0] = 0.0;
			rgb[1] = 0.0;
			rgb[2] = 0.0;

			double intensity = 0;
			// Convolve the kernel at each pixel
			for (int rd = -radius; rd <= radius; ++rd)
			{
				// Get the pixel value
				//QRgb pixel = buffer.pixel(c + rd + radius, r);
				double pixel = buffer[(r + radius)*(w + 2 * radius) + c + rd + radius];
				// Get the value of the kernel
				double weight = kernel[rd + radius];

				intensity += weight*pixel;
			}

			buffer[(r + radius)*(w + 2 * radius) + c + radius] = intensity;
		}
	}

	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double rgb[3];

			rgb[0] = 0.0;
			rgb[1] = 0.0;
			rgb[2] = 0.0;
			double intensity = 0;

			for (int cd = -radius; cd <= radius; ++cd)
			{
				// Get the pixel value
				//QRgb pixel = buffer.pixel(c, r + cd + radius);
				double pixel = buffer[(r + cd + radius)*(w + 2 * radius) + c + radius];

				// Get the value of the kernel
				double weight = kernel[cd + radius];
				intensity += weight*pixel;
			}

			result[r*w + c] = intensity;
		}
	}


	delete[] kernel;

	return result;
    // To access the pixel (c,r), use image[r*width + c].
}


void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
	// Add your code here.  Done right, you should be able to copy most of the code from GaussianBlurImage.

	if (sigma == 0) return;

	if (!image->isGrayscale())
		BlackWhiteImage(image);

	int radius = 3 * sigma;
	int size = 2 * radius + 1;

	QImage buffer;
	int w = image->width();
	int h = image->height();

	buffer = image->copy(-radius, -radius, w + 2 * radius, h + 2 * radius);

	double* kernel = new double[size];

	double Z = sqrt(2 * PI)*sigma;


	for (int rd = -radius; rd <= radius; ++rd)
	{
		kernel[rd + radius] = exp(-(pow(rd, 2.0)) / (2 * pow(sigma, 2))) / Z;
	}

	double denom = 0.000001;
	double y_denom = 0.000001;

	for (int i = 0; i < size; ++i)
	{
		denom += kernel[i];
	}

	for (int i = 0; i < size; i++)
	{
		kernel[i] /= denom;
	}


	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double intensity = 0;
			for (int rd = -radius; rd <= radius; ++rd)
			{
				// Get the pixel value
				QRgb pixel = buffer.pixel(c + radius, r + rd + radius);

				// Get the value of the kernel
				double weight = kernel[rd + radius];
				intensity += weight*(double)qRed(pixel);
			}

			// Store mean pixel in the image to be returned.
			buffer.setPixel(c + radius, r + radius, qRgb((int)floor(intensity + 0.5), (int)floor(intensity + 0.5), (int)floor(intensity + 0.5)));
		}
	}


	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double intensity = 0;
			for (int cd = -radius; cd <= radius; ++cd)
			{
				// Get the pixel value
				QRgb pixel = buffer.pixel(c + cd + radius, r + radius);

				// Get the value of the kernel
				double weight = kernel[cd + radius];
				intensity += weight*(double)qRed(pixel);
			}

			// Store mean pixel in the image to be returned.
			image->setPixel(c, r, qRgb((int)floor(intensity + 0.5), (int)floor(intensity + 0.5), (int)floor(intensity + 0.5)));
		}
	}

	delete[] kernel;

}



void MainWindow::FristDerivate(const vector<double>&image,int w, int h, double sigma, Direction direction, vector<double>& gradient)
{

	if (sigma == 0) return;

	int radius = 1;
	int size = 2 * radius + 1;

	std::vector<double> buffer2((w + 2 * radius)*(h + 2 * radius));

	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			buffer2[(r + radius)*(w + 2 * radius) + c + radius] = image[r*w + c];
		}
	}

	double*x_kernel = new double[size];
	double*y_kernel = new double[size];

	double Z = sqrt(2 * PI)*sigma;
	for (int x = -radius; x <= radius; ++x)
	{
		x_kernel[x + radius] = -x / pow(sigma, 2)*exp(-pow(x, 2.0) / (2 * pow(sigma, 2))) / Z;
	}

	for (int y = -radius; y <= radius; ++y)
	{
		y_kernel[y + radius] = exp(-pow(y, 2.0) / (2 * pow(sigma, 2))) / Z;
	}


	double x_denom = 0.000001;
	double y_denom = 0.000001;

	for (int i = 0; i < size; ++i)
	{
		x_denom += abs(x_kernel[i]);
		y_denom += y_kernel[i];
	}

	for (int i = 0; i < size; i++)
	{
		x_kernel[i] /= x_denom;
		y_kernel[i] /= y_denom;
	}

	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double intensity = 0;
			// Convolve the kernel at each pixel
			for (int cd = -radius; cd <= radius; ++cd)
			{
				double pixel = 0;
				// Get the pixel value
				if (direction == Direction::Y)
					//pixel = buffer.pixel(c, r + cd + radius);
					pixel = buffer2[(r + cd + radius)*(w + 2 * radius) + c + radius];
				else
					//pixel = buffer.pixel(c + cd + radius, r);
					pixel = buffer2[(r + radius)*(w + 2 * radius) + c + cd + radius];

				// Get the value of the kernel
				double weight = y_kernel[cd + radius];
				intensity += weight*pixel;
			}

			buffer2[(r + radius)*(w + 2 * radius) + c + radius] = intensity;

		}
	}


	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			double intensity = 0;

			// Convolve the kernel at each pixel
			for (int rd = -radius; rd <= radius; ++rd)
			{
				// Get the pixel value
				double pixel = 0;
				if (direction == Direction::Y)
					pixel = buffer2[(r + radius)*(w + 2 * radius) + c + rd + radius];
				else
					pixel = buffer2[(r + rd + radius)*(w + 2 * radius) + c + radius];

				// Get the value of the kernel
				double weight = x_kernel[rd + radius];
				intensity += weight*pixel;
			}

			//intensity += 128;
			intensity = floor(intensity + 0.5);
			gradient[r*w+c] = intensity;
		}
	}


	delete[] x_kernel;
	delete[] y_kernel;

}



/*******************************************************************************
Detect Harris corners.
    image - input image
    sigma - standard deviation of Gaussian used to blur corner detector
    thres - Threshold for detecting corners
    interestPts - returned interest points
    numInterestsPts - number of interest points returned
    imageDisplay - image returned to display (for debugging)
*******************************************************************************/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres, CIntPt **interestPts, int &numInterestsPts, QImage &imageDisplay)
{
	/**sift detector*/
	DrawInterestPoints(SiftDetector(image), imageDisplay);
	return;

	/**harris corner detector*/
    int w = image.width();
    int h = image.height();

	vector<double> buffer(w*h);
    QRgb pixel;
    numInterestsPts = 0;
    // Compute the corner response using just the green channel
    for(int r=0;r<h;r++)
       for(int c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Write your Harris corner detection code here.

	vector<double> dx(w*h);
	vector<double> dy(w*h);

	FristDerivate(buffer, w, h, sigma, Direction::X, dx);
	FristDerivate(buffer, w, h, sigma, Direction::Y, dy);

	vector<double> ddx(w*h);
	vector<double> ddy(w*h);
	vector<double> dxdy(w*h);

	for (int r = 0; r < h; r++)
	for (int c = 0; c < w; c++)
	{
		ddx[r*w + c] = dx[r*w + c] * dx[r*w + c];
		ddy[r*w + c] = dy[r*w + c] * dy[r*w + c];
		dxdy[r*w + c] = dx[r*w + c] * dy[r*w + c];
	}


	ddx=SeparableGaussianBlurImage(ddx, w, h, sigma+0.5);
	ddy=SeparableGaussianBlurImage(ddy, w, h, sigma+0.5);
	dxdy=SeparableGaussianBlurImage(dxdy, w, h, sigma+0.5);



	int radius = static_cast<int>(0.02*std::min(w, h) / 2 + 0.5);

	int size = 2 * radius + 1;
	vector<double> r_value((w + 2 * radius)*(h + 2 * radius));

	const double alpha = 0.05;  // constant value 0.04~0.06
	double r_min = 1000000;
	double r_max = 0;
	for (int r = 0; r < h;++r)
	{
		for (int c = 0; c < w;++c)
		{
			double delta = ddx[r*w + c] * ddy[r*w + c] - pow(dxdy[r*w + c], 2);
			double trace = ddx[r*w + c] + ddy[r*w + c];
			double R = delta - alpha*pow(trace, 2);
			if (R>0)
			{
				if (r_max < R) r_max = R;
				if (r_min>R) r_min = R;
				r_value[(r + radius)*(w + 2 * radius) + c + radius] = R;
			}
		}
	}


	thres = r_min + 0.2*(r_max - r_min);

	vector<CIntPt> interest_points;

	int corner_row = 0;
	int corner_col = 0;

	// non maximum suppression
	
	const int LowNum = 1000;
	const int HighNum = 2000;

	int iterations = 0;
	while (!(numInterestsPts>LowNum && numInterestsPts<HighNum) && iterations<10)
	{
		numInterestsPts = 0;
		interest_points.clear();
		for (int r = 0; r < h; ++r)
		{
			for (int c = 0; c < w; ++c)
			{
				double max_value = r_value[(r + radius)*(w + 2 * radius) + c + radius];
				if (max_value>thres)
				{
					for (int rd = -radius; rd <= radius; ++rd)
					for (int cd = -radius; cd <= radius; ++cd)
					{
						if (max_value < r_value[(r + rd + radius)*(w + 2 * radius) + c + cd + radius])
						{
							max_value = r_value[(r + rd + radius)*(w + 2 * radius) + c + cd + radius];
							corner_row = r + rd;
							corner_col = c + cd;
						}
					}

					CIntPt a_interest;
					a_interest.m_X = corner_col;
					a_interest.m_Y = corner_row;
					interest_points.push_back(a_interest);
				}
			}
		}

		numInterestsPts = interest_points.size();
		if (numInterestsPts < LowNum) thres -= thres / 2;
		else if (numInterestsPts>HighNum) thres += thres / 2;

		++iterations;
	}


    // Once you uknow the number of interest points allocate an array as follows:

	numInterestsPts = interest_points.size();
     *interestPts = new CIntPt [numInterestsPts];

    // Access the values using: (*interestPts)[i].m_X = 5.0;
	 for (int i = 0; i < numInterestsPts;++i)
    {
		 (*interestPts)[i].m_X = interest_points[i].m_X;
		 (*interestPts)[i].m_Y = interest_points[i].m_Y;
    }

    // The position of the interest point is (m_X, m_Y)
    // The descriptor of the interest point is stored in m_Desc
    // The length of the descriptor is m_DescSize, if m_DescSize = 0, then it is not valid.

    // Once you are done finding the interest points, display them on the image
    DrawInterestPoints(*interestPts, numInterestsPts, imageDisplay);

}



std::vector<std::pair<int, int>> MainWindow::FindMax(std::vector<double>& level_above, 
	std::vector<double>& level, 
	std::vector<double>& level_below, 
	int width,
	int height, 
	double thresh)
{

	const int Radius = 1;
	vector<pair<int, int>> result;

	for (int r = 0; r < height;++r)
	{
		for (int c = 0; c < width; ++c)
		{
			double extrema_value = level[r*width + c];

			if (extrema_value>thresh)
			{
				bool flag = true;
				for (int rd = -Radius; rd <= Radius; ++rd)
				{
					for (int cd = -Radius; cd <= Radius; ++cd)
					{
						if (!((r + rd) < 0 || (r + rd) >= height || (c + cd) < 0 || (c + cd) >= width))
						{
							double value_above = level_above[(r + rd)*width + c + cd];
							double value = level[(r + rd)*width + c + cd];
							double value_below = level_below[(r + rd)*width + c + cd];

							if (extrema_value<value_above || extrema_value <value || extrema_value < value_below)
							{
								flag = false;
								break;
							}
						}
					}
					if (flag == false) break;
				}

				if (flag) result.push_back(make_pair(r, c));
			}
		}
	}

	return result;
}


std::vector<std::pair<int, int>> MainWindow::FindMin(std::vector<double>& level_above, 
	std::vector<double>& level, 
	std::vector<double>& level_below, 
	int width, 
	int height,
	double thresh)
{
	const int Radius = 1;
	vector<pair<int, int>> result;

	for (int r = 0; r < height; ++r)
	{
		for (int c = 0; c < width; ++c)
		{
			double extrema_value = level[r*width + c];
			if (extrema_value<-thresh)
			{
				bool flag = true;
				for (int rd = -Radius; rd <= Radius; ++rd)
				{
					for (int cd = -Radius; cd <= Radius; ++cd)
					{
						if (!((r + rd) < 0 || (r + rd) >= height || (c + cd) < 0 || (c + cd) >= width))
						{
							double value_above = level_above[(r + rd)*width + c + cd];
							double value = level[(r + rd)*width + c + cd];
							double value_below = level_below[(r + rd)*width + c + cd];

							if (extrema_value>value_above || extrema_value >value || extrema_value>value_below)
							{
								flag = false;
								break;
							}
						}
					}
					if (flag == false) break;
				}

				if (flag) result.push_back(make_pair(r, c));
			}
		}
	}

	return result;
}


vector<tuple<pair<double, double>, double, double, vector<double>>> MainWindow::SiftDetector(QImage image)
{

	
	const double InitialSigma = 0.707107;
	const int S = 2;
	const double K = pow(2, 1.0 / S);
	double sigma =InitialSigma;

	const int GaussianScale = S + 3;
	const int DogScale = GaussianScale-1;
	const int OctaveNum = 3;
	const int Levels = OctaveNum*DogScale;
	const int BorderWidth = 1;

	const double Scale = 2.0;
	double corase_thresh = 3;
	double contrast_thresh = 5;
	double edge_thresh = 10;


	bool upsample = true;

	QImage buffer = image.copy();
	if (!image.isGrayscale())
		BlackWhiteImage(&buffer);

	int image_width = buffer.width();
	int image_height = buffer.height();

	int w =image_width;
	int h = image_height;


	vector<double> original_image(w*h);

	for (int r = 0; r < h; r++)
	for (int c = 0; c < w; c++)
	{
		QRgb pixel = buffer.pixel(c, r);
		for (int i = 0; i < DogScale + 1; ++i)
		{
			original_image[r*w + c] = (double)qGreen(pixel);
		}
	}

	vector<double> image_a(original_image);
	vector<pair<int, int>> image_size(OctaveNum);
	vector<vector<double>> pyramid_dog(Levels);
	vector<vector<double>> pyramid_gaussian(OctaveNum*GaussianScale);
	vector<double> gaussian_scale(OctaveNum*GaussianScale);


	// build pyramid_dog
	if (upsample)
	{
		original_image = std::move(UpSampling(original_image, w, h, Scale));
		original_image = std::move(SeparableGaussianBlurImage(original_image, w, h, 0.5));
	}


	for (int oc = 0; oc < OctaveNum; ++oc)
	{
		for (int gs = 0; gs < GaussianScale; ++gs)
		{
			image_a = std::move(SeparableGaussianBlurImage(original_image, w, h, sigma));

			pyramid_gaussian[oc*GaussianScale + gs] = image_a;

			gaussian_scale[oc*GaussianScale + gs] = sigma;

			sigma *= K;
		}

		for (int ds = 0; ds < DogScale; ++ds)
		{
			pyramid_dog[oc*DogScale + ds].resize(w*h);

			for (int i = 0; i < w*h; ++i)
			{
				pyramid_dog[oc*DogScale + ds][i] = pyramid_gaussian[oc*GaussianScale + ds + 1][i] - pyramid_gaussian[oc*GaussianScale + ds][i];

			}
		}

		image_size[oc] = make_pair(w, h);
		original_image = std::move(DownSampling(original_image, w, h, Scale));

		sigma = gaussian_scale[oc*GaussianScale+S];
	}
	

	// find extrema
	vector<vector<pair<int, int>>> coord_extrema;
	coord_extrema.reserve(OctaveNum*(DogScale - 2));

	for (int oc = 0; oc < OctaveNum; ++oc)
	{
		int width = image_size[oc].first;
		int height = image_size[oc].second;

		for (int ds = 1; ds < DogScale-1; ++ds)
		{
			vector<pair<int, int>> coord_max = FindMax(pyramid_dog[oc*DogScale + ds + 1], 
				pyramid_dog[oc*DogScale + ds], 
				pyramid_dog[oc*DogScale + ds - 1], 
				width, height, corase_thresh);

			vector<pair<int, int>> coord_min = FindMin(pyramid_dog[oc*DogScale + ds + 1], 
				pyramid_dog[oc*DogScale + ds], 
				pyramid_dog[oc*DogScale + ds - 1],
				width, height, corase_thresh);

			coord_max.insert(coord_max.end(), coord_min.begin(), coord_min.end());

			coord_extrema.push_back(std::move(coord_max));
		}	
	}


	//key point localization 
	vector<tuple<pair<double, double>, double, int>>feature0;

	for (int oc = 0; oc < OctaveNum; ++oc)
	{
		int width = image_size[oc].first;
		int height = image_size[oc].second;

		for (int ds = 1; ds < DogScale - 1; ++ds)
		{
			int index = oc*(DogScale - 2) + ds - 1;
			int coord_num = coord_extrema[index].size();

			for (int cn = 0; cn < coord_num; ++cn)
			{
				int rd = coord_extrema[index][cn].first;
				int cd = coord_extrema[index][cn].second;

				vector<pair<int, int>> coord_xy;
				for (int i = -1; i <= 1; ++i)
				for (int j = -1; j <= 1; ++j)
				{
					coord_xy.push_back(make_pair(rd + i, cd + j));
				}

				vector<pair<int, int>> coord_z;
				coord_z.push_back(make_pair(rd - 1, cd));
				coord_z.push_back(make_pair(rd, cd - 1));
				coord_z.push_back(make_pair(rd, cd));
				coord_z.push_back(make_pair(rd, cd + 1));
				coord_z.push_back(make_pair(rd + 1, cd));


				vector<double> dog_k(coord_xy.size());
				vector<double> dog_k_plus(coord_z.size());
				vector<double> dog_k_minus(coord_z.size());

				bool border = false;
				if (rd-1 < BorderWidth || rd+1 >= height - BorderWidth || cd-1 < BorderWidth || cd+1 >= width - BorderWidth)
				{
					border = true;
				}

				if (border) continue;
				else
				{
					for (int i = 0; i < coord_xy.size(); ++i)
					{
						int r = coord_xy[i].first;
						int c = coord_xy[i].second;
						dog_k[i] = pyramid_dog[oc*DogScale + ds][r*width + c];
					}

					for (int i = 0; i < coord_z.size(); ++i)
					{
						int r = coord_z[i].first;
						int c = coord_z[i].second;

						dog_k_plus[i] = pyramid_dog[oc*DogScale + ds + 1][r*width + c];
						dog_k_minus[i] = pyramid_dog[oc*DogScale + ds - 1][r*width + c];
					}

					int error = 0;
					dmat hessian = newdmat(0, 2, 0, 2, &error);
					hessian.el[0][0] = dog_k_plus[2] + dog_k_minus[2] - 2 * dog_k[4];
					hessian.el[1][1] = dog_k[5] + dog_k[3] - 2 * dog_k[4];
					hessian.el[2][2] = dog_k[7] + dog_k[1] - 2 * dog_k[4];
					hessian.el[0][1] = hessian.el[1][0] = ((dog_k_plus[3] - dog_k_minus[3]) - (dog_k_plus[1] - dog_k_minus[1])) / 4;
					hessian.el[0][2] = hessian.el[2][0] = ((dog_k_plus[4] - dog_k_minus[0]) - (dog_k_plus[4] - dog_k_minus[0])) / 4;
					hessian.el[1][2] = hessian.el[2][1] = ((dog_k[8] - dog_k[6]) - (dog_k[2] - dog_k[0])) / 4;

					int error2 = 0;
					dmat gradient_minus = newdmat(0, 2, 0, 0, &error2);
					gradient_minus.el[0][0] = -(dog_k_plus[2] - dog_k_minus[2]) / 2;
					gradient_minus.el[1][0] = -(dog_k[5] - dog_k[3]) / 2;
					gradient_minus.el[2][0] = -(dog_k[7] - dog_k[1]) / 2;

					int error3 = 0;
					dmat x = newdmat(0, 2, 0, 0, &error3);
					int result = solve_system(hessian, x, gradient_minus);

					if (result != 0)
					{
						continue;
					}
					double dog_x = dog_k[4] - (gradient_minus.el[0][0] * x.el[0][0] + gradient_minus.el[1][0] * x.el[1][0] + gradient_minus.el[2][0] * x.el[2][0]) / 2;

					if (abs(x.el[0][0]) > 0.5 || abs(x.el[1][0] > 0.5 || abs(x.el[2][0]) > 0.5))
					{
						continue;
					}
					else if (abs(dog_x) < contrast_thresh)
					{
						continue;
					}
					else
					{
						double trace = hessian.el[1][1] + hessian.el[2][2];
						double delta = hessian.el[1][1] * hessian.el[2][2] - hessian.el[1][2] * hessian.el[1][2];
						double ratio = trace*trace / delta;

						if (ratio < edge_thresh)
							continue;

						//print_mat(x);
						double rd_sub = rd + x.el[2][0]; // y_delta
						double cd_sub = cd + x.el[1][0]; // x_delta

						double scale_sub = gaussian_scale[oc*GaussianScale + ds] + x.el[0][0];
						double scale_dif_a = abs(scale_sub - gaussian_scale[oc*GaussianScale + ds - 1]);
						double scale_dif_b = abs(x.el[0][0]);
						double scale_dif_c = abs(scale_sub - gaussian_scale[oc*GaussianScale + ds + 1]);

						int scale_id = scale_dif_a < scale_dif_b ? (scale_dif_a < scale_dif_c ? ds - 1 : ds + 1) : (scale_dif_b < scale_dif_c ? ds : ds + 1);
						scale_id += oc*GaussianScale;

						feature0.push_back(make_tuple(make_pair(rd_sub, cd_sub), scale_sub, scale_id));
					}

					freemat(hessian);
					freemat(gradient_minus);
					freemat(x);
				}
			}
		}
	}

	//DrawInterestPoints(feature, imageDisplay);


	vector<tuple<pair<double, double>, double, double, int>>feature1;

	// orientation assignment
	int feature_num = feature0.size();
	const int BinNum = 36;
	const double DegreePerBin = 360 / BinNum;
	for (int fe = 0; fe < feature_num;++fe)
	{

		int r = round(get<0>(feature0[fe]).first);
		int c = round(get<0>(feature0[fe]).second);

		auto sigma = get<1>(feature0[fe]);
		auto gs_id = get<2>(feature0[fe]);
		auto radius = round(3 * 1.5*sigma);

		auto size = 2 * radius + 1;
		auto length = size*size;

		// check border
		bool border = false;

		int current_width = image_size[gs_id/GaussianScale].first;
		int current_height = image_size[gs_id / GaussianScale].second;

		if (r - radius < BorderWidth || r + radius >= current_height - BorderWidth || c - radius < BorderWidth || c + radius >= current_width - BorderWidth)
		{
			border = true;
		}

		if (border) continue;
		else
		{

			// computer gradient magnitude and orientation for neighbor pixel 
			vector<double> magnitude(length);
			vector<double>orientation(length);

			for (int rd = -radius,k=0; rd <= radius; ++rd)
			for (int cd = -radius; cd <= radius; ++cd)
			{
				double up = (r + rd - 1)*current_width + c + cd;
				double down = (r + rd + 1)*current_width + c + cd;
				double left = (r + rd)*current_width + c + cd - 1;
				double right = (r + rd)*current_width + c + cd + 1;

				double dx = pyramid_gaussian[gs_id][right] - pyramid_gaussian[gs_id][left];
				double dy = pyramid_gaussian[gs_id][down] - pyramid_gaussian[gs_id][up];

				magnitude[k] = sqrt(dx*dx + dy*dy);

				double angle = 180 * atan(dy / dx) / PI;
				if (angle > 0)
				{
					if (dx > 0) // 1
						orientation[k] = angle;
					else // 3
						orientation[k] = 180 + angle;
				}
				else
				{
					if (dx > 0)  // 4
						orientation[k] = 360 + angle;
					else  //2
						orientation[k] = 180 + angle;
				}

				++k;
			}

			// compute gradient histogram
			array<double, BinNum + 2> hist = {};
			array<double, BinNum > dst = {};
			vector<double> weight(length);
			vector<double> o_bin(length);

			for (int rd = -radius,k=0; rd <= radius; ++rd)
			for (int cd = -radius; cd <= radius; ++cd)
			{
				double gaussian_weight = exp(-(rd*rd + cd*cd) / (2 * (1.5*sigma)*(1.5*sigma)));
				o_bin[k] = orientation[(rd + radius)*size + cd + radius] / DegreePerBin - 0.5;
				weight[k] = magnitude[(rd + radius)*size + cd + radius] * gaussian_weight;
				++k;
			}


			// linear interpolation
			for (int k = 0; k <length;++k)
			{
				double obin = o_bin[k];
				int o0 = floor(obin);
				obin -= o0;

				if (o0 < 0)
					o0 += BinNum;
				if (o0 >= BinNum)
					o0 -= BinNum;

				double v1 = weight[k] * obin;
				double v0 = weight[k] - v1;

				hist[o0] += v0;
				hist[o0+1] += v1;
			}
			hist[1] += hist[1+BinNum];

			// copy 
			for (int b = 0; b < BinNum;++b)
			{
				dst[b] = hist[b + 1];
			}

			// smooth histogram
			int iteration_times = 6;
			for (int n = 0; n < iteration_times;++n)
			{
				for (int i = 1; i < BinNum - 1; ++i)
				{
					dst[i] = (dst[i - 1] + dst[i] + dst[i + 1]) / 3;
				}
			}
			

			double hist_max = 0;
			double hist_second = 0;
			int id_max = 0;
			int id_second = 0;
			for (int i = 0; i < BinNum;++i)
			{
				if (hist_max < dst[i])
				{
					hist_second = hist_max;
					hist_max = dst[i];
					id_max = id_second = i;
				}
				else if (hist_second < dst[i])
				{
					hist_second = dst[i];
					id_second = i;
				}
			}

			feature1.push_back(make_tuple(get<0>(feature0[fe]), sigma, id_max*DegreePerBin, gs_id));

			if (hist_second>0.8*hist_max 
				&& hist_second > dst[std::max(0, id_second - 1)]
				&& hist_second > dst[std::min(BinNum - 1, id_second + 1)])
			{	
				feature1.push_back(make_tuple(get<0>(feature0[fe]), sigma, id_second*DegreePerBin, gs_id));
			}

		}

	}

	// sift descriptor
	feature_num = feature1.size();
	const int BinNum2 = 8;
	const double BinPerDegree = BinNum2/360.0;
	const int Bp = 4;
	const double exp_scale = 1.0 / (Bp*Bp*0.5); 
	const double SiftDescrMagThr = 0.2;
	const double SiftIntDescrFctr = 512.0;

	vector<tuple<pair<double, double>, double, double, vector<double>>>feature_final; // position scale orientation descriptor

	for (int fe = 0; fe < feature_num; ++fe)
	{
		int row = round(get<0>(feature1[fe]).first);
		int col = round(get<0>(feature1[fe]).second);
		auto sigma = get<1>(feature1[fe]);
		auto angle = get<2>(feature1[fe]);
		auto gs_id = get<3>(feature1[fe]);

		int radius = round((3 * sigma*sqrt(2.0)*(Bp + 1) + 1) / 2);
		int length = (2 * radius + 1)*(2*radius+1);
		int descriptor_length = Bp*Bp*BinNum2;

		double bin_width = 3 * sigma;
		double gaussian = bin_width*Bp / 2;

		vector<double> mag(length);
		vector<double> orientation(length);
		vector<double> weight(length);
		vector<double> r_bin(length);
		vector<double> c_bin(length);
		vector<double> hist((Bp+2)*(Bp+2)*(BinNum2+2));
		vector<double> dst(descriptor_length);

		// check border
		bool border = false;

		int current_width = image_size[gs_id / GaussianScale].first;
		int current_height = image_size[gs_id / GaussianScale].second;


		//double diagonal = sqrt(2.0)*radius;
		if (row - radius < BorderWidth || row + radius >= current_height - BorderWidth || col - radius < BorderWidth || col + radius >= current_width - BorderWidth)
		{
			border = true;
		}
		if (border) continue;
		else 
		{
			double radians = angle*PI / 180;  // 
			int id = 0;
			// rotate to main orientation
			for (int rd = -radius; rd <= radius; ++rd)
			for (int cd = -radius; cd <= radius; ++cd)
			{
				double c_rot = cd*cos(radians) - rd*sin(radians);
				double r_rot = cd*sin(radians) + rd*cos(radians);	

				// divide neighbor pixel into four bins
				double rbin = r_rot / bin_width + Bp / 2 - 0.5;
				double cbin = c_rot / bin_width + Bp / 2 - 0.5;
				if (rbin >-1 && rbin < Bp && cbin>-1 && cbin < Bp)
				{
					double up = (row + rd - 1)*current_width + col + cd;
					double down = (row + rd + 1)*current_width + col + cd;
					double left = (row + rd)*current_width + col + cd - 1;
					double right = (row + rd)*current_width + col + cd + 1;

					double dx = pyramid_gaussian[gs_id][right] - pyramid_gaussian[gs_id][left];
					double dy = pyramid_gaussian[gs_id][down] - pyramid_gaussian[gs_id][up];

					r_bin[id] = rbin;
					c_bin[id] = cbin;
					mag[id] = sqrt(dx*dx + dy*dy);
					weight[id] = exp(-(r_rot*r_rot + c_rot*c_rot)*exp_scale/ (2 * gaussian*gaussian));

					double angle = 180 * atan(dy / dx) / PI;
					if (angle > 0)
					{
						if (dx > 0) // 1
							orientation[id] = angle;
						else // 3
							orientation[id] = 180 + angle;
					}
					else
					{
						if (dx > 0)  // 4
							orientation[id] = 360 + angle;
						else  //2
							orientation[id] = 180 + angle;
					}

					++id;
				}
			}

			// compute histogram 
			for (int k = 0; k < id; ++k)
			{
				double rbin = r_bin[k];
				double cbin = c_bin[k];

				double ori_diff = orientation[k] - angle;
				if (ori_diff < 0)
					ori_diff += 360;
				double obin = ori_diff*BinPerDegree;
				double vk = mag[k] * weight[k];

				int r0 = floor(rbin);
				int c0 = floor(cbin);
				int o0 = floor(obin);

				rbin -= r0;
				cbin -= c0;
				obin -= o0;

				// trilinear interpolation
				double v1 = vk*rbin, v0 = vk - v1;
				double v11 = v1*cbin, v10 = v1*(1-cbin);
				double v01 = v0*cbin, v00 = v0*(1-cbin);
				double v111 = v11*obin, v110 = v11*(1 - obin);
				double v101 = v10*obin, v100 = v10*(1 - obin);
				double v011 = v01*obin, v010 = v01*(1 - obin);
				double v001 = v00*obin, v000 = v00*(1 - obin);

				int idx0 = ((r0 + 1)*(Bp + 2) + c0 + 1)*(BinNum2 + 2) + o0;
				int idx1 = ((r0 + 1)*(Bp + 2) + c0 + 2)*(BinNum2 + 2) + o0;
				int idx2 = ((r0 + 2)*(Bp + 2) + c0 + 1)*(BinNum2 + 2) + o0;
				int idx3 = ((r0 + 2)*(Bp + 2) + c0 + 2)*(BinNum2 + 2) + o0;


				hist[idx0] += v000;
				hist[idx0+1] += v001;
				hist[idx1]+= v010;
				hist[idx1 + 1] += v011;
				hist[idx2] = v100;
				hist[idx2 + 1] += v101;
				hist[idx3] = v110;
				hist[idx3 + 1] += v111;

			}


			// every 8 bin should be a circle 
			for (int i = 0; i < Bp;++i)
			for (int j = 0; j < Bp;++j)
			{
				int idx = ((i + 1)*(Bp + 2) + j + 1)*(BinNum2 + 2);
				hist[idx] += hist[idx + BinNum2];
				hist[idx + 1] += hist[idx + BinNum2 + 1];
				for (int k = 0; k < BinNum2;++k)
				{
					dst[(i*Bp+j)*BinNum2+k] = hist[idx + k];
				}
			}
			
			// apply hysteresis thresholding and scale the result so that it can be easily to convert to byte array
			double norm2 = 0;
			for (int k = 0; k < descriptor_length;++k)
			{
				norm2 += dst[k]*dst[k];
			}
			double thr = SiftDescrMagThr*sqrt(norm2);

			norm2 = 0;
			for (int i = 0; i < descriptor_length; ++i)
			{
				double val = std::min(thr, dst[i]);
				dst[i] = val;
				norm2 += val*val;
			}
			norm2 = SiftIntDescrFctr / std::max<double>(std::sqrt(norm2), FLT_EPSILON);
			
			for_each(begin(dst), end(dst), [norm2](double& e){e = saturate_cast<uchar>(e*norm2); });

			double scl = (double)image_width / current_width;
			double real_row = get<0>(feature1[fe]).first*scl;
			double real_col = get<0>(feature1[fe]).second*scl;

			feature_final.push_back(make_tuple(make_pair(real_row, real_col), sigma, angle, dst));
		}

	}

	return feature_final;
}



/*******************************************************************************
Find matching interest points between images.
    image1 - first input image
    interestPts1 - interest points corresponding to image 1
    numInterestsPts1 - number of interest points in image 1
    image2 - second input image
    interestPts2 - interest points corresponding to image 2
    numInterestsPts2 - number of interest points in image 2
    matches - set of matching points to be returned
    numMatches - number of matching points returned
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::MatchInterestPoints(QImage image1, CIntPt *interestPts1, int numInterestsPts1,
                             QImage image2, CIntPt *interestPts2, int numInterestsPts2,
                             CMatches **matches, int &numMatches, QImage &image1Display, QImage &image2Display)
{
    numMatches = 0;

    // Compute the descriptors for each interest point.
    // You can access the descriptor for each interest point using interestPts1[i].m_Desc[j].
    // If interestPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
  //  ComputeDescriptors(image1, interestPts1, numInterestsPts1);
  //  ComputeDescriptors(image2, interestPts2, numInterestsPts2);

	vector<tuple<pair<double, double>, double, double, vector<double>>>dscriptor0=SiftDetector(image1);
	vector<tuple<pair<double, double>, double, double, vector<double>>>dscriptor1=SiftDetector(image2);

    // Add your code here for finding the best matches for each point.
	int interest_num0 = dscriptor0.size();
	int interest_num1 = dscriptor1.size();
	int feature_num = get<3>(dscriptor0[0]).size();
	const double RatioThresh = 0.75;

	vector<pair<double, double>> feature0;
	vector<pair<double, double>> feature1;


	for (int i = 0; i < interest_num0;++i)
	{
		vector<double> f0 = get<3>(dscriptor0[i]);

		double dmin = 1000000;
		double dsecond = 1000000;
		int i2 = 0;
		for (int j = 0; j < interest_num1;++j)
		{
			vector<double> f1 = get<3>(dscriptor1[j]);

			double d = 0;
			for (int k = 0; k < feature_num;++k)
			{
				d += (f1[k] - f0[k])*(f1[k] - f0[k]);
			}
			if (dmin > d)
			{
				dsecond = dmin;
				dmin = d;	
				i2 = j;
			}
			else if (dsecond>d)
			{
				dsecond = d;
			}
		}
		
		if (dmin < RatioThresh*dsecond)
		{			
			feature0.push_back(get<0>(dscriptor0[i]));
			feature1.push_back(get<0>(dscriptor1[i2]));
		}

	}

	DrawMatches(feature0, feature1, image1Display, image2Display);

}

void MainWindow::MatchInterestPoints(QImage image1, 
	QImage image2, 
	std::vector<std::pair<double,double>>& feature0,
	std::vector<std::pair<double, double>>& feature1,
	QImage &image1Display,
	QImage &image2Display)
{

	// Compute the descriptors for each interest point.
	// You can access the descriptor for each interest point using interestPts1[i].m_Desc[j].
	// If interestPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
	//  ComputeDescriptors(image1, interestPts1, numInterestsPts1);
	//  ComputeDescriptors(image2, interestPts2, numInterestsPts2);

	vector<tuple<pair<double, double>, double, double, vector<double>>>dscriptor0 = SiftDetector(image1);
	vector<tuple<pair<double, double>, double, double, vector<double>>>dscriptor1 = SiftDetector(image2);

	// Add your code here for finding the best matches for each point.
	int interest_num0 = dscriptor0.size();
	int interest_num1 = dscriptor1.size();
	int feature_num = get<3>(dscriptor0[0]).size();
	const double RatioThresh = 0.75;

	for (int i = 0; i < interest_num0; ++i)
	{
		vector<double> f0 = get<3>(dscriptor0[i]);

		double dmin = 1000000;
		double dsecond = 1000000;
		int i2 = 0;
		for (int j = 0; j < interest_num1; ++j)
		{
			vector<double> f1 = get<3>(dscriptor1[j]);

			double d = 0;
			for (int k = 0; k < feature_num; ++k)
			{
				d += (f1[k] - f0[k])*(f1[k] - f0[k]);
			}
			if (dmin > d)
			{
				dsecond = dmin;
				dmin = d;
				i2 = j;
			}
			else if (dsecond > d)
			{
				dsecond = d;
			}
		}

		if (dmin < RatioThresh*dsecond)
		{
			feature0.push_back(get<0>(dscriptor0[i]));
			feature1.push_back(get<0>(dscriptor1[i2]));
		}

	}

	//DrawMatches(feature0, feature1, image1Display, image2Display);
}



/*******************************************************************************
Project a point (x1, y1) using the homography transformation h
    (x1, y1) - input point
    (x2, y2) - returned point
    h - input homography used to project point
*******************************************************************************/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3])
{
    // Add your code here.

	x2 =(h[0][0] * x1 + h[0][1] * y1 + h[0][2])/(h[2][0]*x1+h[2][1]*y1+h[2][2]);
	y2 = (h[1][0] * x1 + h[1][1] * y1 + h[1][2]) / (h[2][0] * x1 + h[2][1] * y1 + h[2][2]);

}

/*******************************************************************************
Count the number of inliers given a homography.  This is a helper function for RANSAC.
    h - input homography used to project points (image1 -> image2
    matches - array of matching points
    numMatches - number of matchs in the array
    inlierThreshold - maximum distance between points that are considered to be inliers

    Returns the total number of inliers.
*******************************************************************************/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold)
{
    // Add your code here.
	int inlier = 0;
	for (int i = 0; i < numMatches;++i)
	{
		double  x_project = 0;
		double y_project = 0;
		Project(matches[i].m_X1, matches[i].m_Y1, x_project, y_project, h);
		double diff = sqrt((matches[i].m_X2 - x_project)*(matches[i].m_X2 - x_project) +
			(matches[i].m_Y2 - y_project)*(matches[i].m_Y2 - y_project));
		if (diff < inlierThreshold)
			++inlier;
	}

    return inlier;
}


int MainWindow::ComputeInlierCount(double h[3][3], 
	std::vector<std::pair<double,double>>& feature0, 
	std::vector<std::pair<double, double>>& feature1,
	double inlierThreshold)
{
	// Add your code here.
	int inlier = 0;
	int num_match = feature0.size();
	for (int i = 0; i < num_match; ++i)
	{
		double  x_project = 0;
		double y_project = 0;
		Project(feature0[i].second, feature0[i].first, x_project, y_project, h);
		double diff = sqrt((feature1[i].second - x_project)*(feature1[i].second - x_project) +
			(feature1[i].first - y_project)*(feature1[i].first - y_project));

		if (diff < inlierThreshold)
			++inlier;
	}

	return inlier;
}



/*******************************************************************************
Compute homography transformation between images using RANSAC.
    matches - set of matching points between images
    numMatches - number of matching points
    numIterations - number of iterations to run RANSAC
    inlierThreshold - maximum distance between points that are considered to be inliers
    hom - returned homography transformation (image1 -> image2)
    homInv - returned inverse homography transformation (image2 -> image1)
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
                        double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display)
{
    // Add your code here.

	vector<pair<double, double>> feature0;
	vector<pair<double, double>> feature1;


	const int PairNum = 4;
	int feature_num = feature0.size();
	int score = 0;
	double hom_best[3][3];
	for (int it = 0; it < numIterations; ++it)
	{
		default_random_engine random((unsigned)time(0));
		uniform_int_distribution<unsigned> generate(0, feature_num);

		vector<pair<double, double>> pairs0(PairNum);
		vector<pair<double, double>> pairs1(PairNum);

		for (int i = 0; i < PairNum; ++i)
		{
			int id = generate(random);
			pairs0[i] = feature0[id];
			pairs1[i] = feature1[id];
		}
		ComputeHomography(pairs0, pairs1, hom, true);
		int num = ComputeInlierCount(hom, feature0, feature1, inlierThreshold);
		if (score < num)
		{
			score = num;
			hom_best[0][0] = hom[0][0];
			hom_best[0][1] = hom[0][1];
			hom_best[0][2] = hom[0][2];
			hom_best[1][0] = hom[1][0];
			hom_best[1][1] = hom[1][1];
			hom_best[1][2] = hom[1][2];
			hom_best[2][0] = hom[2][0];
			hom_best[2][1] = hom[2][1];
			hom_best[2][2] = hom[2][2];
		}
	}


	int num_match = feature0.size();
	vector<pair<double, double>> pairs0;
	vector<pair<double, double>> pairs1;
	for (int i = 0; i < num_match; ++i)
	{
		double  x_project = 0;
		double y_project = 0;
		Project(feature0[i].second, feature0[i].first, x_project, y_project, hom);
		double diff = sqrt((feature1[i].second - x_project)*(feature1[i].second - x_project) +
			(feature1[i].first - y_project)*(feature1[i].first - y_project));

		if (diff < inlierThreshold)
		{
			pairs0.push_back(feature0[i]);
			pairs1.push_back(feature1[i]);
		}
	}

	ComputeHomography(pairs0, pairs1, hom, true);
	ComputeHomography(pairs0, pairs1, homInv, false);


	DrawMatches(feature0, feature1, image1Display, image2Display);

    // After you're done computing the inliers, display the corresponding matches.
    //DrawMatches(inliers, numInliers, image1Display, image2Display);

}

void MainWindow::RANSAC(std::vector<std::pair<double, double>>& feature0,
	std::vector<std::pair<double, double>>& feature1,
	int numIterations,
	double inlierThreshold,
	double hom[3][3],
	double homInv[3][3],
	QImage &image1Display,
	QImage &image2Display)
{
	// Add your code here.
	// a:
	// randomly choose 4 pair matches
	// use these matches to compute homography
	// used computed homography count inliers
	// repeat inliers  reach to highest number, such homography as best one
	// b: recompute homography with all inliers, not just 4.
	const int PairNum = 4;
	int feature_num = feature0.size();
	int score = 0;
	double hom_best[3][3];

	auto is_distinct = [](int a, int b[], int i)->bool{
		for (int j = 0; j < i; ++j)
		{
			if (a == b[j])
			{
				return false;
			}
		}
		return true;
	};

	default_random_engine random((unsigned)time(0));
	uniform_int_distribution<unsigned> generate(0, feature_num - 1);

	for (int it = 0; it < numIterations;++it)
	{	
		vector<pair<double, double>> pairs0(PairNum);
		vector<pair<double, double>> pairs1(PairNum);

		int ids[PairNum];
		for (int i = 0; i < PairNum;++i)
		{
			int r = 0;
		    do 
		    {
				r = generate(random);
			} while (!is_distinct(r, ids, i));		
			ids[i] = r;
		}

		for (int i = 0; i < PairNum; ++i)
		{
			pairs0[i] = feature0[ids[i]];
			pairs1[i] = feature1[ids[i]];
		}

		ComputeHomography(pairs0, pairs1, hom, true);
		int num=ComputeInlierCount(hom, feature0, feature1, inlierThreshold);
		if (score < num)
		{
			score = num;
			hom_best[0][0] = hom[0][0];
			hom_best[0][1] = hom[0][1];
			hom_best[0][2] = hom[0][2];
			hom_best[1][0] = hom[1][0];
			hom_best[1][1] = hom[1][1];
			hom_best[1][2] = hom[1][2];
			hom_best[2][0] = hom[2][0];
			hom_best[2][1] = hom[2][1];
			hom_best[2][2] = hom[2][2];
		}
	}


	int num_match = feature0.size();
	vector<pair<double, double>> pairs0;
	vector<pair<double, double>> pairs1;
	for (int i = 0; i < num_match; ++i)
	{
		double  x_project = 0;
		double y_project = 0;
		Project(feature0[i].second, feature0[i].first, x_project, y_project, hom_best);
		double diff = sqrt((feature1[i].second - x_project)*(feature1[i].second - x_project) +
			(feature1[i].first - y_project)*(feature1[i].first - y_project));

		if (diff < inlierThreshold)
		{
			pairs0.push_back(feature0[i]);
			pairs1.push_back(feature1[i]);
		}
	}

	ComputeHomography(pairs0, pairs1, hom, true);
	ComputeHomography(pairs0, pairs1, homInv, false);
	

	DrawMatches(pairs0, pairs1, image1Display, image2Display);

	// After you're done computing the inliers, display the corresponding matches.
	//DrawMatches(inliers, numInliers, image1Display, image2Display);
}




/*******************************************************************************
Bilinearly interpolate image (helper function for Stitch)
    image - input image
    (x, y) - location to interpolate
    rgb - returned color values

    You can just copy code from previous assignment.
*******************************************************************************/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    // Add your code here.
	double w = image->width();
	double h = image->height();

	//if (x <= 0 || x >= w - 1 || y <= 0 || y >= h - 1)
	//{
	//	rgb[0] = 0;
	//	rgb[1] = 0;
	//	rgb[2] = 0;
	//	return false;
	//}

	int x_left = static_cast<int>(x);
	int x_right = x_left + 1;
	int y_left = static_cast<int>(y);
	int y_right = y_left + 1;

	//if (image->isGrayscale())
	//{
	//	QRgb pixel = image->pixel(x_left, y_left);
	//	double f00;
	//	f00 = static_cast<double>(qRed(pixel));


	//	pixel = image->pixel(x_right, y_left);
	//	double f10;
	//	f10= static_cast<double>(qRed(pixel));

	//	pixel = image->pixel(x_left, y_right);
	//	double f01;
	//	f01 = static_cast<double>(qRed(pixel));


	//	pixel = image->pixel(x_right, y_right);
	//	double f11;
	//	f11 = static_cast<double>(qRed(pixel));

	//	double r0 = (x_right - x)*f00 + (x - x_left)*f10;
	//	double r1 = (x_right - x)*f01 + (x - x_left)*f11;
	//	rgb[0] = rgb[1] = rgb[2] = (y_right - y)*r0 + (y - y_left)*r1;
	//}

	//else
	//{
		QRgb pixel = image->pixel(x_left, y_left);
		double f00[3];
		f00[0] = static_cast<double>(qRed(pixel));
		f00[1] = static_cast<double>(qGreen(pixel));
		f00[2] = static_cast<double>(qBlue(pixel));

		pixel = image->pixel(x_right, y_left);
		double f10[3];
		f10[0] = static_cast<double>(qRed(pixel));
		f10[1] = static_cast<double>(qGreen(pixel));
		f10[2] = static_cast<double>(qBlue(pixel));

		pixel = image->pixel(x_left, y_right);
		double f01[3];
		f01[0] = static_cast<double>(qRed(pixel));
		f01[1] = static_cast<double>(qGreen(pixel));
		f01[2] = static_cast<double>(qBlue(pixel));

		pixel = image->pixel(x_right, y_right);
		double f11[3];
		f11[0] = static_cast<double>(qRed(pixel));
		f11[1] = static_cast<double>(qGreen(pixel));
		f11[2] = static_cast<double>(qBlue(pixel));


		for (int i = 0; i < 3; ++i)
		{
			double r0 = (x_right - x)*f00[i] + (x - x_left)*f10[i];
			double r1 = (x_right - x)*f01[i] + (x - x_left)*f11[i];
			rgb[i] = (y_right - y)*r0 + (y - y_left)*r1;
		}
	//}

    return true;
}


/*******************************************************************************
Stitch together two images using the homography transformation
    image1 - first input image
    image2 - second input image
    hom - homography transformation (image1 -> image2)
    homInv - inverse homography transformation (image2 -> image1)
    stitchedImage - returned stitched image
*******************************************************************************/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage)
{
    // Width and height of stitchedImage
    int ws = 0;
    int hs = 0;

    // Add your code to compute ws and hs here.
	int w1 = image1.width();
	int h1 = image1.height();
	int w2 = image2.width();
	int h2 = image2.height();

	double x_lu = 0;
	double y_lu = 0;

	double x_ru = w2-1;
	double y_ru = 0;

	double x_ld = 0;
	double y_ld = h2-1;

	double x_rd = w2-1;
	double y_rd = h2-1;

	double x_project[4];
	double y_project[4];

	Project(x_lu, y_lu, x_project[0], y_project[0], homInv);
	Project(x_ru, y_ru, x_project[1], y_project[1], homInv);
	Project(x_ld, y_ld, x_project[2], y_project[2], homInv);
	Project(x_rd, y_rd, x_project[3], y_project[3], homInv);

	sort(begin(x_project), end(x_project));
	sort(begin(y_project), end(y_project));

	ws = std::max(w1, round(x_project[3])) - std::min(0, round(x_project[0]));
	hs = std::max(h1, round(y_project[3])) - std::min(0, round(y_project[0]));

	struct bbx
	{
		double up;
		double down;
		double left;
		double right;
	}project =
	{
		std::min(std::min(y_project[0], y_project[1]), std::min(y_project[2], y_project[3])),
		std::max(std::max(y_project[0], y_project[1]), std::max(y_project[2], y_project[3])),
		std::min(std::min(x_project[0], x_project[1]), std::min(x_project[2], x_project[3])),
		std::max(std::max(x_project[0], x_project[1]), std::max(x_project[2], x_project[3])),
	};

	bbx intersect =
	{
		0,
		0,
		0,
		0
	};
	// compute intersect region (image1 intersect with image2 projected on image1)
	int pos = -1;
	for (int i = 0; i < 4;++i)
	{
		if (0 < project.right && 0 > project.left && 0 < project.down && 0 > project.up)
		{
			pos = 0;
			break;
		}

		if (w1 < project.right && w1 > project.left && 0 < project.down && 0 > project.up)
		{
			pos = 1;
			break;
		}

		if (0 < project.right && 0 > project.left && h1 < project.down && h1 > project.up)
		{
			pos = 2;
			break;
		}

		if (w1 < project.right && w1 > project.left && h1 < project.down && h1 > project.up)
		{
			pos = 3;
			break;
		}
	}


	if (pos == 0 || pos==2) // (0,0)
	{
		intersect.up = project.up;
		intersect.down = project.down;
		intersect.left = 0;
		intersect.right = project.right;
	}
	if (pos == 1 || pos==3) //(0,w1) 
	{
		intersect.up = project.up;
		intersect.down = project.down;
		intersect.left = project.left;
		intersect.right = w1;

	}

	double tx = abs(std::min(0.0, x_project[0]));
	double ty = abs(std::min(0.0, y_project[0]));

    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

	// copy image1 to stitched image
	for (int r = 0; r < h1;++r)
	for (int c = 0; c < w1;++c)
	{
		QRgb pixel = image1.pixel(c, r);
		int r2 = round(r + ty);
		int c2 = round(c + tx);
		stitchedImage.setPixel(c2, r2, pixel);
	}

	// dynamic find optimum seam
	// status transform function is: d[i][j]={e[i][j]+min{d[i-1][j-1],d[i-1][j],d[i-1][j+1]}}

	vector<int> seam;

	int cx = floor(intersect.left);
	int ry = floor(intersect.up);

	int rows = floor(intersect.down) - floor(intersect.up) + 1;
	int cols = floor(intersect.right) - floor(intersect.left) + 1;

	// energy function is:  e[i][j]= sqrt(I2(i,j)*I2(i,j)-I1(i,j)*I1(i,j))

	vector<vector<double>> distance(rows,vector<double>(cols));
	vector<vector<int>> table(rows,vector<int>(cols));
	const double Infinity = 1.0e6;

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			double r2 = 0;
			double c2 = 0;

			int r1 = r + ry;
			int c1 = c + cx;

			Project(c1, r1, c2, r2, hom);  // sift and project

			double rgb1[3];
			double rgb2[3];
			memset(rgb1, 0, sizeof(rgb1));
			memset(rgb2, 0, sizeof(rgb2));

			bool flag = false;
			if (r1>=0 && r1<=h1-1 && c1>=0 && c1<=w1-1)
			if (r2>=0 && r2<=h2 - 1 && c2>=0 && c2 <= w2 - 1) // within image2 region
			{
				QRgb pixel = image2.pixel(c2,r2);
				rgb1[0] = static_cast<double>(qRed(pixel));
				rgb1[1] = static_cast<double>(qGreen(pixel));
				rgb1[2] = static_cast<double>(qBlue(pixel));

				pixel = image1.pixel(c1, r1);
				rgb2[0] = static_cast<double>(qRed(pixel));
				rgb2[1] = static_cast<double>(qGreen(pixel));
				rgb2[2] = static_cast<double>(qBlue(pixel));
				flag = true;
			}
			
			// compute energy 
			double energy = 0;
			if (flag)
			{
				for (int i = 0; i < 3; ++i)
				{
					energy += sqrt((rgb2[i] - rgb1[i])*(rgb2[i] - rgb1[i]));
				}
			}
			else
			{
				energy = Infinity;
			}

			if (r == 0)
			{
				distance[r][c] = energy;
			}
			else
			{
				if (c == 0)
				{
					distance[r][c] = energy + std::min(distance[r - 1][c], distance[r - 1][c + 1]);
					table[r][c] = distance[r - 1][c] < distance[r - 1][c + 1] ? 0 : 1;
				}
				else if (c == cols - 1)
				{
					distance[r][c] = energy + std::min(distance[r - 1][c - 1], distance[r - 1][c]);
					table[r][c] = distance[r - 1][c - 1] < distance[r - 1][c] ? -1 : 0;
				}
				else
				{
					distance[r][c] = energy + std::min(std::min(distance[r - 1][c - 1], distance[r - 1][c]), distance[r - 1][c + 1]);

					table[r][c] = distance[r - 1][c - 1] < distance[r - 1][c] ?
						(distance[r - 1][c - 1] < distance[r - 1][c + 1] ? -1 : 1) : (distance[r - 1][c] < distance[r - 1][c + 1] ? 0 : 1);				
				}
			}
		}
	}

	// find optimum seam
	auto it = min_element(begin(distance[rows - 1]), end(distance[rows - 1]));
	int col = it - begin(distance[rows - 1]);

	seam.push_back(col + cx);
	for (int r = rows - 2; r >=0; --r)
	{
		int d = table[r + 1][col];
		col += d;
		seam.push_back(col + cx);
	}

	// interpolate image2 to stitched image
	for (int r = 0; r < hs;++r)
	for (int c = 0; c < ws;++c)
	{
		QRgb pixel = stitchedImage.pixel(c, r);
		double color[3];
		color[0] = qRed(pixel);
		color[1] = qGreen(pixel);
		color[2] = qBlue(pixel);

		double r2 = 0;
		double c2 = 0;

		int c_shift = c - tx;
		int r_shift = r - ty;

		Project(c_shift, r_shift, c2, r2, hom);  // sift and project
		double rgb[3];
		memset(rgb, 0, sizeof(rgb));

		int radius = 5;
		double alpha = 0.5;

		// is this pixel in intersect region
		if (c_shift< cx + cols && c_shift>=cx-1 && r_shift<ry + rows && r_shift>=ry)
		{
			int sr = rows - 1 + ry - r_shift; 
			if (c_shift<seam[sr]) // lie on left of the seam
			{
				;
			}
			else// lie on right of the seam
			{
				if (r2>=0 && r2<=h2 - 1 && c2>=0 && c2 <= w2 - 1) // within image2 region,
				{
					BilinearInterpolation(&image2, c2, r2, rgb);
					color[0] = rgb[0];
					color[1] = rgb[1];
					color[2] = rgb[2];
				}
			}
			if (c_shift >= seam[sr] - radius && c_shift<=seam[sr]+radius)
			{
				double rgb1[3];
				double rgb2[3];
				memset(rgb1, 0, sizeof(rgb1));
				memset(rgb2, 0, sizeof(rgb2));

				if (r_shift >= 0 && r_shift <= h1 - 1 && c_shift >= 0 && c_shift <= w1 - 1)
				if (r2 >= 0 && r2 <= h2 - 1 && c2 >= 0 && c2 <= w2 - 1) // within image2 region,
				{
					BilinearInterpolation(&image2, c2, r2, rgb);
					pixel = image1.pixel(c_shift, r_shift);
					rgb2[0] = static_cast<double>(qRed(pixel));
					rgb2[1] = static_cast<double>(qGreen(pixel));
					rgb2[2] = static_cast<double>(qBlue(pixel));
	
					color[0] = alpha*rgb[0] + (1 - alpha)*rgb2[0];
					color[1] = alpha*rgb[1] + (1 - alpha)*rgb2[1];
					color[2] = alpha*rgb[2] + (1 - alpha)*rgb2[2];
				}


			}

		}
		else 
		{
			if (r2 >=0 && r2<=h2 - 1 && c2>=0 && c2 <= w2 - 1) // within image2 region,
			{
				BilinearInterpolation(&image2, c2, r2, rgb);				
				color[0] = rgb[0];
				color[1] = rgb[1];
				color[2] = rgb[2];
			}
		}

		stitchedImage.setPixel(c, r, qRgb((int)color[0], (int)color[1], (int)color[2]));
	}


	//// display seam
	//for (int i = 0; i < seam.size();++i)
	//{
	//	int sr = rows - 1 + ry - i; // i=0 correspondence to the last row in intersect region

	//	// then shift them to stitched image coord
	//	sr += ty;
	//	seam[i] += tx;
	//	stitchedImage.setPixel(seam[i], sr, qRgb(255, 0, 0));
	//}
	
}

