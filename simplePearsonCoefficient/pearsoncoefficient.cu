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

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <pearsoncoefficient_kernel.cu>
#include "reduction_kernel.cu"
#include "reduction.h"

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
template<class T> T reduceCPU(T*data, int size);

extern "C"
void computeGold( float* reference, float* idata, const unsigned blocks, const unsigned int len);

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

    // unsigned int N = 1 << 24; // 1 << 24 will produce a NaN on GT330M Cuda Toolkit 3.2
	unsigned int N = 1 << 8;
	unsigned int num_of_threads = 512;
    unsigned int mem_size = sizeof( float) * N;

    // allocate host memory to represent vectors v1 & v2
    float* h_idata = (float*) malloc( mem_size);
    float* h_jdata = (float*) malloc( mem_size);
    float* h_powidata = (float*) malloc( mem_size);
    float* h_powjdata = (float*) malloc( mem_size);
    float* h_aggdata = (float*) malloc( mem_size);

    // initalize the memory
	float t;
    for( unsigned int i = t = 0, t = pow((float)i, 2); i < N; ++i, t=pow((float)i,2)) 
    {
        h_idata[i] = (float) i;
        h_jdata[i] = (float) i;
        h_powidata[i] = (float) t;
        h_powjdata[i] = (float) t;
		h_aggdata[i] = (float)(h_idata[i]*h_jdata[i]);
    }

    // allocate device memory to represent vectors v1 & v2
    float* d_idata;
    float* d_jdata;
    float* d_powidata;
    float* d_powjdata;
    float* d_aggdata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_jdata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_powidata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_powjdata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_aggdata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_idata, h_jdata, mem_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_powidata, h_powidata, mem_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_powidata, h_powjdata, mem_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_aggdata, h_aggdata, mem_size, cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    float* d_o2data;
    float* d_powodata;
    float* d_powo2data;
    float* d_aggodata;
	unsigned int blocks = N%num_of_threads == 0? N/num_of_threads: 1+N/num_of_threads;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, sizeof(float)*blocks ));
    cutilSafeCall( cudaMalloc( (void**) &d_o2data, sizeof(float)*blocks ));
    cutilSafeCall( cudaMalloc( (void**) &d_powodata, sizeof(float)*blocks ));
    cutilSafeCall( cudaMalloc( (void**) &d_powo2data, sizeof(float)*blocks ));
    cutilSafeCall( cudaMalloc( (void**) &d_aggodata, sizeof(float)*blocks ));

	// Conduct parallel sum reduction
	reduce<float>(N, num_of_threads, blocks, 6, d_idata, d_odata);
	reduce<float>(N, num_of_threads, blocks, 6, d_jdata, d_o2data);
	reduce<float>(N, num_of_threads, blocks, 6, d_powidata, d_powodata);
	reduce<float>(N, num_of_threads, blocks, 6, d_powjdata, d_powo2data);
	reduce<float>(N, num_of_threads, blocks, 6, d_aggdata, d_aggodata);

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( sizeof(float)*blocks );
    float* h_o2data = (float*) malloc( sizeof(float)*blocks );
    float* h_powodata = (float*) malloc( sizeof(float)*blocks );
    float* h_powo2data = (float*) malloc( sizeof(float)*blocks );
    float* h_aggodata = (float*) malloc( sizeof(float)*blocks );

    // copy result from device to host
    cutilSafeCallNoSync( cudaMemcpy( h_odata, d_odata, sizeof(float)*blocks , cudaMemcpyDeviceToHost) );
    cutilSafeCallNoSync( cudaMemcpy( h_o2data, d_o2data, sizeof(float)*blocks , cudaMemcpyDeviceToHost) );
    cutilSafeCallNoSync( cudaMemcpy( h_powodata, d_powodata, sizeof(float)*blocks , cudaMemcpyDeviceToHost) );
    cutilSafeCallNoSync( cudaMemcpy( h_powo2data, d_powo2data, sizeof(float)*blocks , cudaMemcpyDeviceToHost) );
    cutilSafeCallNoSync( cudaMemcpy( h_aggodata, d_aggodata, sizeof(float)*blocks , cudaMemcpyDeviceToHost) );

	// compute reference soln 
	// computeGold(h_odata, h_idata, blocks, N);
	// computeGold(h_odata, h_jdata, blocks, N);
	// computeGold(h_powodata, h_powidata, blocks, N);
	// computeGold(h_powodata, h_powjdata, blocks, N);

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

	float sum1 = reduceCPU<float>(h_odata, blocks);
	float sum2 = reduceCPU<float>(h_o2data, blocks);

	float sum1sq = reduceCPU<float>(h_powodata, blocks);
	float sum2sq = reduceCPU<float>(h_powo2data, blocks);

	float psum = reduceCPU<float>(h_aggodata, blocks);
	float num = psum - (sum1*sum2)/N;
	float den = sqrt((sum1sq-pow(sum1,2)/N)*(sum2sq-pow(sum2,2)/N));
	if (den == 0)
		printf("Den is zero\n");
	else
		printf("Den is %f\n", 1.0 - num/den);

    // cleanup memory
    free( h_idata);
    free( h_jdata);
    free( h_odata);
    free( h_o2data);
    free( h_powidata);
    free( h_powjdata);
    free( h_powodata);
    free( h_powo2data);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilSafeCall(cudaFree(d_jdata));
    cutilSafeCall(cudaFree(d_o2data));
    cutilSafeCall(cudaFree(d_powidata));
    cutilSafeCall(cudaFree(d_powodata));
    cutilSafeCall(cudaFree(d_powjdata));
    cutilSafeCall(cudaFree(d_powo2data));

    cudaThreadExit();
}
