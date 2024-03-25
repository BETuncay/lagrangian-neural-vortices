#include "mex.h"
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <fstream>

void doWork(char *exportPath, float* pData, float* pDomainMin, float* pDomainMax, int* pResolution, int* pNumComponents)
{
    // Write header
    {
        std::ofstream outStream(exportPath);
        outStream << "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1" << std::endl;
        outStream << std::endl;
        outStream << std::endl;

        outStream << "define Lattice " << pResolution[0] << " " << pResolution[1] << " " << pResolution[2] << std::endl;
        outStream << std::endl;

        outStream << "Parameters {" << std::endl;
        switch (pNumComponents[0])
        {
            case 1:
                outStream << "Content \"" << pResolution[0] << "x" << pResolution[1] << "x" << pResolution[2] << " float, uniform coordinates\"," << std::endl;
                break;
            case 2:
                outStream << "Content \"" << pResolution[0] << "x" << pResolution[1] << "x" << pResolution[2] << " float[2], uniform coordinates\"," << std::endl;
                break;
            case 3:
                outStream << "Content \"" << pResolution[0] << "x" << pResolution[1] << "x" << pResolution[2] << " float[3], uniform coordinates\"," << std::endl;
                break;
        }

        outStream << "\tBoundingBox " << pDomainMin[0] << " " << pDomainMax[0] << " " << pDomainMin[1] << " " << pDomainMax[1] << " " << pDomainMin[2] << " " << pDomainMax[2] << "," << std::endl;
        outStream << "\tCoordType \"uniform\"" << std::endl;
        outStream << "}" << std::endl;
        outStream << std::endl;

        switch (pNumComponents[0])
        {
            case 1:
                outStream << "Lattice { float Data } @1" << std::endl;
                break;
            case 2:
                outStream << "Lattice { float[2] Data } @1" << std::endl;
                break;
            case 3:
                outStream << "Lattice { float[3] Data } @1" << std::endl;
                break;
        }
        outStream << std::endl;

        outStream << "# Data section follows" << std::endl;
        outStream << "@1" << std::endl;

        outStream.close();
    }

    // Write data
    {
#ifdef WIN32
        std::string dirPath = exportPath.substr(0, std::max(int(exportPath.find_last_of("\\")), int(exportPath.find_last_of("/"))) + 1);
        CreateDirectoryA(dirPath.c_str(), NULL);
#endif
        std::ofstream outStream(exportPath, std::ios::out | std::ios::app | std::ios::binary);
        outStream.write((char*)pData, sizeof(float) * pResolution[0] * pResolution[1] * pResolution[2] * pNumComponents[0]);
        outStream.close();
    }
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("pmInverseCDF2d:nrhs","Six inputs required.");
        return;
    }
    
    if (nlhs != 0) {
        mexErrMsgIdAndTxt("pmInverseCDF2d:nlhs","No output required.");
        return;
    }
    
	/* Input arguments */

    // Get the length of the input string.
    size_t buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;
    // Allocate memory for input and output strings.
    char* importPath = (char*)mxCalloc(buflen, sizeof(char));
    int status = mxGetString(prhs[0], importPath, buflen);
    if (status != 0) 
        mexWarnMsgTxt("Not enough space. String is truncated.");

    float* pData = (float*) mxGetPr(prhs[1]);
    float* pDomainMin = (float*) mxGetPr(prhs[2]);
    float* pDomainMax = (float*) mxGetPr(prhs[3]);
    int* pResolution = (int*) mxGetPr(prhs[4]);
    int* pNumComponents = (int*) mxGetPr(prhs[5]);
    
    doWork(importPath, pData, pDomainMin, pDomainMax, pResolution, pNumComponents);
    
    mxFree(importPath);
}