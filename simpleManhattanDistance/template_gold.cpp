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
//
#include <math.h>
#include <stdio.h>
#include <limits>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"  {
double computeGold( float* pdata, float* qdata, const unsigned int len);

bool compare(double referenceSoln, double* data, unsigned int len);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! For all elements in the two sets, compute (p - q)^2 
//! @param pdata  reference data set 'P', preallocated
//! @param qdata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
double
computeGold(float* pdata, float* qdata, const unsigned int len) 
{
	double temp1 = 0.0;
    for( unsigned int i = 0; i < len; ++i) 
    {
        temp1 += fabs(pdata[i] - qdata[i]);
    }
	return temp1;
}

bool compare(double referenceSoln, double* data, unsigned int len) {
		double temp1 = 0.0;
		for( unsigned int i = 0; i < len; ++i ) {
			temp1 += data[i];
		}
		printf("ref:%e, proposed: %e, epsilon:%e\n", referenceSoln, temp1, std::numeric_limits<double>::epsilon());
		double diff = fabs(referenceSoln - temp1);
		if ( diff < std::numeric_limits<double>::epsilon())   {
			return true;
			printf("(( %e %e", std::numeric_limits<double>::epsilon(), diff);
		}
		else {
			printf(")) %e %e", std::numeric_limits<double>::epsilon(), diff);
				return false;
		}
}

