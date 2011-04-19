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

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include <device_functions.h>
#include <math_functions.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! 2D euclidean space to compute manhattan distance 
//! @param g_pdata  input data in (device)global memory
//! @param g_qdata  output data in (device)global memory
//! @param g_outdata  output partial data in (device)global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_pdata, float* g_qdata, double* g_outdata) 
{
  // shared memory
  __shared__  float cache[threadsPerBlock];

  // load data in device to shared memory
  const unsigned int cacheIndex = threadIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  float temp = 0;
  while ( tid < N ) {
   // perform the computation of |p - q|
   temp += fabs(g_pdata[threadIdx.x] - g_qdata[threadIdx.x]);
   tid += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = temp;
 
  __syncthreads(); // remember its really a memory fence for a warp of threads i.e. warpSize = 32

  // reduction phase
  int i = blockDim.x/2;
  while( i != 0) {
   if (cacheIndex < i) {
    cache[cacheIndex] += cache[cacheIndex + 1];
   }
   __syncthreads();// becareful where u place __syncthreads ...
   i /= 2;
  }
  
  //
  if (cacheIndex == 0)
   g_outdata[blockIdx.x] = cache[0];
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
