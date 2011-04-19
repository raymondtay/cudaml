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
#include "reduction.h"

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Pearson Coefficient Kernel in Python 
////////////////////////////////////////////////////////////////////////////////
//def pearson(v1, v2):
//	sum1=sum(v1)
//	sum2=sum(v2)
//
//	sum1sq=sum([pow(v,2) for v in v1])
//	sum2sq=sum([pow(v,2) for v in v2])
//
//	psum=sum([v1[i]*v2[i] for i in range(len(v1))])
//	num=psum-(sum1*sum2/len(v1))
//	den=sqrt((sum1sq-pow(sum1,2)/len(v1))*(sum2sq-pow(sum2,2)/len(v1)))
//	if den==0: return 0
//	return 1.0-num/den
//
////////////////////////////////////////////////////////////////////////////////
//! Pearson Coefficient Kernel in CUDA 
////////////////////////////////////////////////////////////////////////////////
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
#define DATA_SIZE 256

__global__ void
testKernel( float* g_v1data,float* g_v2data, float* g_odata) 
{
  // shared memory
  __shared__  float sdata[DATA_SIZE];

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation
  SDATA(tid) = g_v1data[tid];
  __syncthreads();

  // perform some computations
  SDATA(tid) = (float) num_threads * SDATA( tid);
  __syncthreads();

  // write data to global memory
  g_odata[tid] = SDATA(tid);
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
