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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cutil_inline.h>

// user defined includes
#define imin(a, b) (a<b?a:b)

const int N = 1 * (1 << 20);
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

// includes, kernels
#include <template_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" {
double computeGold( float* pdata, float* qdata, const unsigned int len);
bool compare(float referenceSoln, float* data, unsigned int len);
}
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
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int mem_size = sizeof( float) * N;

    // allocate host memory for the 'P' & 'Q' dimensions
    float* h_Pdata = (float*) malloc( mem_size);
    float* h_Qdata = (float*) malloc( mem_size);

    // initalize the memory to random numbers
	srand(time(NULL));
    for( unsigned int i = 0; i < N; ++i) 
    {
	   // using http://linux.die.net/man/3/rand as a reference for generating random 
       // numbers
       h_Pdata[i] = 1 + (int)(99.0 * (rand()/ (RAND_MAX + 1.0)));
    }

    for( unsigned int i = 0; i < N; ++i) 
    {
	   // using http://linux.die.net/man/3/rand as a reference for generating random 
       // numbers
       h_Qdata[i] = 1 + (int)(99.0 * (rand()/ (RAND_MAX + 1.0)));
    }


    // allocate device memory for array 'P'
    float* d_Pdata;
    cutilSafeCall( cudaMalloc( (void**) &d_Pdata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_Pdata, h_Pdata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for array 'Q'
    float* d_Qdata;
    cutilSafeCall( cudaMalloc( (void**) &d_Qdata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_Qdata, h_Qdata, mem_size,
                                cudaMemcpyHostToDevice) );

	// allocate device memory for result
	double* d_outdata;

    cutilSafeCall( cudaMalloc( (void**) &d_outdata, blocksPerGrid*4));

    printf("Launching kernel with %d blocks, %d threads-per-block\n", blocksPerGrid, threadsPerBlock);

    // execute the kernel
    testKernel<<< blocksPerGrid, threadsPerBlock >>>( d_Pdata, d_Qdata, d_outdata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

	// Wait til computation is complete
	cudaThreadSynchronize();

    // allocate mem for the result on host side
    float* h_outdata = (float*) malloc( blocksPerGrid*4);

    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_outdata, d_outdata, blocksPerGrid * sizeof(float),
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
	double referSoln = computeGold( h_Pdata, h_Qdata, N);

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat",
                                      h_outdata, N, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        // CUTBoolean res = cutComparef( referenceSoln, h_outdata, N);
		bool res = compare(referSoln, h_outdata, N);
        printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    free( h_Pdata);
    free( h_Qdata);
    free( h_outdata);
    cutilSafeCall(cudaFree(d_Pdata));
    cutilSafeCall(cudaFree(d_Qdata));
    cutilSafeCall(cudaFree(d_outdata));

    cudaThreadExit();
}
