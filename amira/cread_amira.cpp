#include "mex.h"
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>

static const char* FindAndJump(const char* buffer, const char* SearchString)
{
    const char* FoundLoc = strstr(buffer, SearchString);
    if (FoundLoc) return FoundLoc + strlen(SearchString);
    return buffer;
}

template <typename T>
static T ToNumber(const std::string &Text) {
    std::stringstream ss(Text);
    T result;
    return ss >> result ? result : 0;
}

void doWork(char *importPath, mxArray** plhs, float* pDomainMin, float* pDomainMax, int* pResolution, int* pNumComponents)
{
    const bool verbose = false;

    FILE* fp = fopen(importPath, "rb");
    if (!fp)
    {
        printf(("AmiraLoader: Could not find " + std::string(importPath) + "\n").c_str());
        return;
    }

    if (verbose) printf(("AmiraLoader: Reading " + std::string(importPath) + "\n").c_str());

    //We read the first 2k bytes into memory to parse the header.
    //The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    char buffer[2048];
    fread(buffer, sizeof(char), 2047, fp);
    buffer[2047] = '\0'; //The following string routines prefer null-terminated strings

    if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
    {
        printf("AmiraLoader: Not a proper amira file.");
        fclose(fp);
        return;
    }

    //Find the Lattice definition, i.e., the dimensions of the uniform grid
    int xDim(0), yDim(0), zDim(0);
    sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);
    if (verbose) printf("AmiraLoader: Grid Dimensions: %i %i %i", xDim, yDim, zDim);

    pResolution[0] = xDim;
    pResolution[1] = yDim;
    pResolution[2] = zDim;
    
    //Find the BoundingBox
    float xmin(1.0f), ymin(1.0f), zmin(1.0f);
    float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);
    sscanf(FindAndJump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
    if (verbose) printf("AmiraLoader: BoundingBox in x-Direction: [%f ... %f]\n", xmin, xmax);
    if (verbose) printf("AmiraLoader: BoundingBox in y-Direction: [%f ... %f]\n", ymin, ymax);
    if (verbose) printf("AmiraLoader: BoundingBox in z-Direction: [%f ... %f]\n", zmin, zmax);

    pDomainMin[0] = xmin;
    pDomainMin[1] = ymin;
    pDomainMin[2] = zmin;
    
    pDomainMax[0] = xmax;
    pDomainMax[1] = ymax;
    pDomainMax[2] = zmax;
    
    //Is it a uniform grid? We need this only for the sanity check below.
    const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
    if (verbose)
        if (bIsUniform)
            printf("AmiraLoader: GridType: uniform");
        else printf("AmiraLoader: GridType: UNKNOWN");

    //Type of the field: scalar, vector
    int NumComponents(0);
    bool isFloat = true;
    if (strstr(buffer, "Lattice { float Data }"))
    {
        //Scalar field
        NumComponents = 1;
    }
    else
    {
        //A field with more than one component, i.e., a vector field
        if (sscanf(FindAndJump(buffer, "Lattice { float["), "%d", &NumComponents) == 1) isFloat = true;
        else if (sscanf(FindAndJump(buffer, "Lattice { double["), "%d", &NumComponents) == 1) isFloat = false;
    }
    if (verbose) printf("AmiraLoader: Number of Components: %i\n", NumComponents);

    pNumComponents[0] = NumComponents;
    
    //Sanity check
    if (xDim <= 0 || yDim <= 0 || zDim <= 0
        || xmin > xmax || ymin > ymax || zmin > zmax
        || !bIsUniform || NumComponents <= 0)
    {
        printf("AmiraLoader: Something went wrong");
        fclose(fp);
        return;
    }
    
    /**
    if (NumComponents != 3)
    {
        printf("Amira Loader: Only 3D vector fields are supported!\n");
        fclose(fp);
        return;
    }
    */
    
    //Find the beginning of the data section
    const long idxStartData = (long)(strstr(buffer, "# Data section follows") - buffer);
    if (idxStartData > 0)
    {
        //Set the file pointer to the beginning of "# Data section follows"
        fseek(fp, idxStartData, SEEK_SET);
        //Consume this line, which is "# Data section follows"
        fgets(buffer, 2047, fp);
        //Consume the next line, which is "@1"
        fgets(buffer, 2047, fp);

        //Read the data
        // - how much to read
        const size_t NumToRead = xDim * yDim * zDim * NumComponents;

        if (isFloat)
        {
            plhs[0] = mxCreateNumericMatrix(xDim * yDim * zDim * NumComponents, 1, mxSINGLE_CLASS, mxREAL); // data
            float* fltData = (float*) mxGetData(plhs[0]);
            const size_t ActRead = fread((void*)fltData, sizeof(float), NumToRead, fp);
            if (NumToRead != ActRead)
            {
                printf("AmiraLoader: Something went wrong while reading the binary data section.\nPremature end of file?");
                fclose(fp);
                return;
            }
        }
        else
            printf("AmiraLoader: Double not implemented!\n");
    }
}

// nlhs, mxArray erg, rest input
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("pmInverseCDF2d:nrhs","One input required.");
        return;
    }
    
    if (nlhs != 5) {
        mexErrMsgIdAndTxt("pmInverseCDF2d:nlhs","Five outputs required.");
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

    
    /* Output arguments */
    
    plhs[1] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL); // min corner
    plhs[2] = mxCreateNumericMatrix(3, 1, mxSINGLE_CLASS, mxREAL); // max corner
    plhs[3] = mxCreateNumericMatrix(3, 1, mxINT32_CLASS, mxREAL); // resolution
    plhs[4] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL); // number of components

    float* pDomainMin = (float*) mxGetData(plhs[1]);
    float* pDomainMax = (float*) mxGetData(plhs[2]);
    int* pResolution = (int*) mxGetData(plhs[3]);
    int* pNumComponents = (int*) mxGetData(plhs[4]);
    
    doWork(importPath, plhs, pDomainMin, pDomainMax, pResolution, pNumComponents);
    
    mxFree(importPath);
}