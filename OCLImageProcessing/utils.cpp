#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cmath>

#include "utils.h"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::string;
using std::ifstream;

string FileToString(const string fileName)
{
	ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

	try
	{
		size_t size;
		char*  str;
		string s;

		if (f.is_open())
		{
			size_t fileSize;
			f.seekg(0, ifstream::end);
			size = fileSize = f.tellg();
			f.seekg(0, ifstream::beg);

			str = new char[size + 1];
			if (!str) throw(string("Could not allocate memory"));

			f.read(str, fileSize);
			f.close();
			str[size] = '\0';

			s = str;
			delete[] str;
			return s;
		}
	}
	catch (std::string msg)
	{
		cerr << "Exception caught in FileToString(): " << msg << endl;
		if (f.is_open())
			f.close();
	}
	catch (...)
	{
		cerr << "Exception caught in FileToString()" << endl;
		if (f.is_open())
			f.close();
	}
	string errorMsg = "FileToString()::Error: Unable to open file "
		+ fileName;
	throw(errorMsg);
}

void createGaussianFilter(float * kernel, int size, float sd)
{
	float s;
	s = 2.0f * (float)M_PI * sd * sd;

	// set sum for normalization
	float sum = 0.0f;

	// generate a size by size matrix
	int centre = size / 2;

	for (int x = -centre; x <= centre; x++)
	{
		for (int y = -centre; y <= centre; y++)
		{
			int idx = (x + centre) * size + (y + centre);
			kernel[idx] = (expf(-(x * x + y * y) / 2)) / s;
			sum += kernel[idx];
		}
	}

	// normalize the vector
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			kernel[i * size + j] /= sum;
		}
	}

}
