
// Copyright (c) 2010-2015, Raymond Tay, Singapore
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <cudpp/cudpp.h>


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
void computeSumScanGold( float *reference, const float *idata, 
                        const unsigned int len,
                        const CUDPPConfiguration &config);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    cutilDeviceInit(argc, argv);

    unsigned int numElements = 32768;
	size_t xpitch;
	size_t ypitch;
	unsigned int height;
	unsigned int width = height = sqrt(numElements);
    unsigned int memSize = sizeof( float) * numElements;

    // allocate host memory which represents X n Y respectively
    float* h_idata = (float*) malloc( memSize);
    float* h_jdata = (float*) malloc( memSize);

    // initalize the memory
    for (unsigned int i = 0; i < numElements; ++i) 
    {
        h_idata[i] = (float) (rand() & 0xf);
        h_jdata[i] = (float) (rand() & 0xf);
    }

    // allocate device memory; in my case i have 4 arrays
    float* d_idata; // for storing X^2
    float* d_jdata; // for storing Y^2
    float* d_kdata; // for storing 2xy
    float* d_ldata; // for storing

	// xpitch & ypitch should be equal....i think 
    cutilSafeCall( cudaMallocPitch( (void**) &d_idata, &xpitch, sizeof(float)* width, sizeof(float)* height));
    cutilSafeCall( cudaMallocPitch( (void**) &d_jdata, &ypitch, sizeof(float)* width, sizeof(float)* height));

    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, memSize,
                                cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_jdata, h_jdata, memSize,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, memSize));

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult result = cudppPlan(&scanplan, config, numElements, 1, xpitch);  

    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    cudppScan(scanplan, d_odata, d_idata, numElements);

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, memSize,
                                cudaMemcpyDeviceToHost) );
    // compute reference solution
    // float* reference = (float*) malloc( memSize);
    // computeSumScanGold( reference, h_idata, numElements, config);

    // check result
    // CUTBoolean res = cutComparef( reference, h_odata, numElements);
    // printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    result = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }
    
    free( h_idata);
    free( h_odata);
    // free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
}
