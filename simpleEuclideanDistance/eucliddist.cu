//
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

#include <thrust/host_vector.h>
#if __CUDA_ARCH__ == 200 
 #include <thrust/device_vector.h>
#elif __CUDA_ARCH__ == 100
 #include <thrust/device_ptr.h>
#endif
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

struct powfunctor
{
	__host__ __device__
	double operator()(const float& p, const float& q) const {
		return pow( (float)(p - q), 2);
	}
	//double operator()(const float& p, const float& q) const {
	//	return p+q;
	//}
};
// gold solution to compute x + y
template <typename InputIterator1, typename InputIterator2>
double computeGold_2(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2) 
{
	double sum = 0.0;
	for(; (first1 != last1) && (first2 != last2); ++first1, ++first2) {
		sum += *first1 + *first2;
	}

	std::cout << "Gold=" << sum << std::endl;	

	return sum;
}

template <typename InputIterator1, typename InputIterator2>
double computeGold_1(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2) 
{
	float sum = 0.0;
	for(; (first1 != last1) && (first2 != last2); ++first1, ++first2) {
		sum += pow((float)(*first1 - *first2), 2);
	}

	float s1 = sqrt(sum);
	std::cout << "Gold=" << s1 << std::endl;	

	return s1;
}

int main(void)
{
#if __CUDA_ARCH__ == 200
	thrust::device_vector<float> p_vec(1 << 20);
	thrust::device_vector<float> q_vec(1 << 20);
	thrust::device_vector<float> r_vec(1 << 20);
	thrust::generate(p_vec.begin(), p_vec.end(), rand);
	thrust::generate(q_vec.begin(), q_vec.end(), rand);
	// Current Thrust's transformations supports 2 input vectors, so we use it
	thrust::transform(p_vec.begin(), p_vec.end(), q_vec.begin(), r_vec.begin(), powfunctor());

	float sum = thrust::reduce(r_vec.begin(), r_vec.end(), (int)0, thrust::plus<float>());
	std::cout << "sqrt(" << sum  << ")=" << sqrt(sum) << std::endl;	
// #elif __CUDA_ARCH__ == 100
#else
	unsigned int N = 1 << 20;
	thrust::host_vector<float> p_vec(N);
	thrust::host_vector<float> q_vec(N);
	thrust::host_vector<float> r_vec(N);

	srand(0);
	thrust::generate(p_vec.begin(), p_vec.end(), rand);
	thrust::generate(q_vec.begin(), q_vec.end(), rand);

	float referenceSoln = computeGold_1(p_vec.begin(), p_vec.end(), q_vec.begin(), q_vec.end());

	// device memory 'raw' pointers
	float* raw_ptr_P;
	float* raw_ptr_Q;
	float* raw_ptr_R;

	cudaMalloc( (void**)&raw_ptr_P, (N)*sizeof(float));
	cudaMalloc( (void**)&raw_ptr_Q, (N)*sizeof(float));
	cudaMalloc( (void**)&raw_ptr_R, (N)*sizeof(float));

	thrust::device_ptr<float> dev_ptr_P(raw_ptr_P);
	thrust::device_ptr<float> dev_ptr_Q(raw_ptr_Q);
	thrust::device_ptr<float> dev_ptr_R(raw_ptr_R);

	thrust::copy(p_vec.begin(), p_vec.end(), dev_ptr_P);
	thrust::copy(q_vec.begin(), q_vec.end(), dev_ptr_Q);

	// uncommenting the following will produce errors for 1.x devices 
	// complaining that CUDA doesn't support function pointers and function 
	// templates. reason is because a host function like 'rand' cannot be 
	// executed in the device i.e. GPU
	//thrust::generate(dev_ptr_P, dev_ptr_Q + N, rand);
	//thrust::generate(dev_ptr_Q, dev_ptr_Q + N, rand);

	thrust::transform(dev_ptr_P, dev_ptr_P + N, dev_ptr_Q, dev_ptr_R, powfunctor());
	float sum = thrust::reduce(dev_ptr_R, dev_ptr_R + N, (float)0, thrust::plus<float>());

	std::cout << "1. GPU " << sqrt(sum) << std::endl;	
	std::cout << "2. CPU " << referenceSoln << std::endl;	
#endif

	std::cout << "END" << std::endl;
	return 0;
}

