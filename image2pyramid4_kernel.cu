#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <fstream>
#include <string>
#include <cub/cub.cuh>
struct Avgpool_pyramid4_params
{
    int n, c, h, w;
	int h_tiles, w_tiles, tiles;
	int r2_h_tiles, r2_w_tiles;
	int r4_h_tiles, r4_w_tiles;
	float *output_0;
	float *output_2;
	float *output_4;
	float *output_8;
	float const *input;
	int32_t input_batch_stride;
	int32_t input_c_stride;

	
	int32_t output_0_batch_stride;
	int32_t output_2_batch_stride;
	int32_t output_4_batch_stride;
	int32_t output_8_batch_stride;
};

template<int32_t kNWarps=4>
__global__ void avgpool_pyramid4_kernel(Avgpool_pyramid4_params params){
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tile_id = (blockIdx.x * kNWarps) + tidy;
	int tile_hid = tile_id / params.w_tiles;
	int tile_wid = tile_id - tile_hid * params.w_tiles;
	int tile_offset = blockIdx.y * params.input_batch_stride + 
						tile_hid * params.w * 8 +
						tile_wid * 8; // offset of 8x8 tile
	int tid = tidx;
	int g4 = tid >> 4;
	tid -= (g4 << 4);
	int g3 = tid >> 3;
	tid -= (g3 << 3);
	int g2 = tid >> 2;
	tid -= (g2 << 2);
	int g1 = tid >> 1;
	tid -= (g1 << 1);
	int g0 = tid;
	// tile_offset + g4*4*w + g3*4 + g2*2*w + g1*2 + g0*w;
	int inputIdx = tile_offset + ((g4<<2) + (g2<<1) + g0) * params.w + (g3<<2) + (g1<<1);
	float2 val2;
	float2 read2;

	read2 = *reinterpret_cast<float2 const *>(&params.input[inputIdx]);
	val2.x = read2.x * 0.299; val2.y = read2.y * 0.299;
	read2 = *reinterpret_cast<float2 const *>(&params.input[inputIdx + params.input_c_stride]);
	val2.x += read2.x * 0.587; val2.y += read2.y * 0.587;
	read2 = *reinterpret_cast<float2 const *>(&params.input[inputIdx + (params.input_c_stride<<1)]);
	val2.x += read2.x * 0.114; val2.y += read2.y * 0.114;

	// write back to 1x1, same size
	int outputIdx = inputIdx + blockIdx.y * (params.output_0_batch_stride-params.input_batch_stride);
	*reinterpret_cast<float2 *>(&params.output_0[outputIdx]) = val2;

	// reduce for 2x2
	float val = val2.x + val2.y;
	int output_2_idx = blockIdx.y * params.output_2_batch_stride + 
					(tile_hid << 2) * params.r2_w_tiles + (tile_wid << 2) + 
					g4 * 2 * params.r2_w_tiles +
					g3 * 2 + 
					g2 * params.r2_w_tiles +
					g1;
	val = val + __shfl_xor_sync(uint32_t(-1), val, 1);
	val /= 4;
	if((tidx & 1) == 0){
		params.output_2[output_2_idx] = val;
	}


	int output_4_idx = blockIdx.y * params.output_4_batch_stride + 
					(tile_hid << 1) * params.r4_w_tiles + (tile_wid << 1) + 
					g4 * params.r4_w_tiles +
					g3;
	val = val + __shfl_xor_sync(uint32_t(-1), val, 2);
	val = val + __shfl_xor_sync(uint32_t(-1), val, 4);
	val /= 4;
	if((tidx & 7) == 0){
		params.output_4[output_4_idx] = val;
	}

	int output_8_idx = blockIdx.y * params.output_8_batch_stride + 
						tile_id;
	val = val + __shfl_xor_sync(uint32_t(-1), val, 8);
	val = val + __shfl_xor_sync(uint32_t(-1), val, 16);
	val /= 4;
	if(tidx == 0){
		params.output_8[output_8_idx] = val;
	}
}

std::vector<at::Tensor> avgpool_pyramid4_forward_cuda(
	at::Tensor x
){
	int32_t batch_size = x.size(0);
    int32_t c = x.size(1);
    int32_t h = x.size(2);
    int32_t w = x.size(3);

	auto output_0 = at::empty({batch_size, 1, h, w}, x.options());
	auto output_2 = at::empty({batch_size, 1, h/2, w/2}, x.options());
	auto output_4 = at::empty({batch_size, 1, h/4, w/4}, x.options());
	auto output_8 = at::empty({batch_size, 1, h/8, w/8}, x.options());

	// setup params
	Avgpool_pyramid4_params params;
	params.n = batch_size;
	params.c = c;
	params.h = h;
	params.w = w;
	params.h_tiles = (h / 8); params.w_tiles = (w / 8);
	params.tiles = params.h_tiles * params.w_tiles;
	params.output_8_batch_stride = params.tiles;

	params.r4_h_tiles = h / 4; params.r4_w_tiles = w / 4; 
	params.output_4_batch_stride = params.r4_h_tiles * params.r4_w_tiles;

	params.r2_h_tiles = h / 2; params.r2_w_tiles = w / 2; 
	params.output_2_batch_stride = params.r2_h_tiles * params.r2_w_tiles;

	params.input_batch_stride = c * h * w;
	params.input_c_stride = h * w;
	params.output_0_batch_stride = params.input_c_stride;
	params.input =  reinterpret_cast<float const *>(x.data_ptr<float>());
	params.output_0 =  reinterpret_cast<float *>(output_0.data_ptr<float>());
	params.output_2 =  reinterpret_cast<float *>(output_2.data_ptr<float>());
	params.output_4 =  reinterpret_cast<float *>(output_4.data_ptr<float>());
	params.output_8 =  reinterpret_cast<float *>(output_8.data_ptr<float>());

	constexpr int kNWarps = 4;

	dim3 grid;
	dim3 block;
	grid.x = params.tiles/kNWarps; grid.y = params.n; grid.z = 1; 
	block.x = 32; block.y = kNWarps; block.z = 1;
	avgpool_pyramid4_kernel<kNWarps><<<grid, block>>>(params);
	return {output_0, output_2, output_4, output_8};
}
















































struct Cross_accum_params
{
    int32_t n, c, h, w, hw;
	
	float const *input;
	float *output;

	int32_t input_batch_stride;
	int32_t output_batch_stride;
};

template<int32_t ThrRepeat, int32_t ElementsPerBlock>
__global__ void cross_accum_kernel(Cross_accum_params params){
	int32_t tidx = threadIdx.x;
	int32_t bidx = blockIdx.x;

	int32_t batch_offset = blockIdx.y * params.input_batch_stride;
	int32_t imageIdx = bidx * ElementsPerBlock + (tidx << 1);
	if(imageIdx >= params.hw){
		return;
	}
	int32_t inputGmemIdx = batch_offset + imageIdx;
	float2 input2, other2, grad_a2, grad_b2, grad2;
	float w_pre, w_post;
	input2 = *reinterpret_cast<float2 const*>(&params.input[inputGmemIdx]);

	int32_t hidx = imageIdx / params.w;
	int32_t widx = imageIdx - hidx*params.w;
	
	int32_t outputGmemIdx = blockIdx.y * params.output_batch_stride + 
							imageIdx;


	// horizontal, along w axis
	w_post = widx + 2 < params.w ? params.input[inputGmemIdx + 2] : 0.;
	grad_a2.x = input2.y; grad_a2.y = w_post;

	w_pre = widx > 0 ? params.input[inputGmemIdx - 1] : 0.;
	grad_a2.x -= w_pre; grad_a2.y -= input2.x;

	*reinterpret_cast<float2 *>(&params.output[outputGmemIdx]) = grad_a2;


	// vertical, along h axis
	other2 = hidx + 1 < params.h ? *reinterpret_cast<float2 const*>(&params.input[inputGmemIdx + params.w]) : make_float2(0., 0.);
	grad_b2.x = other2.x; grad_b2.y = other2.y; 

	other2 = hidx > 0 ? *reinterpret_cast<float2 const*>(&params.input[inputGmemIdx - params.w]) : make_float2(0., 0.);
	grad_b2.x -= other2.x; grad_b2.y -= other2.y;

	*reinterpret_cast<float2 *>(&params.output[outputGmemIdx + params.hw]) = grad_b2;


	// combine gradient of 2 axis
	grad2.x = sqrtf(grad_a2.x*grad_a2.x + grad_b2.x*grad_b2.x);
	grad2.y = sqrtf(grad_a2.y*grad_a2.y + grad_b2.y*grad_b2.y);

	*reinterpret_cast<float2 *>(&params.output[outputGmemIdx + (params.hw << 1)]) = grad2;
}

at::Tensor cross_accum_forward_cuda(
	at::Tensor x
){
	int32_t batch_size = x.size(0);
    int32_t h = x.size(2);
    int32_t w = x.size(3);

	auto output_tensor = at::empty({batch_size, 3, h, w}, x.options());

	constexpr int32_t thr_rep = 1;
	constexpr int32_t nThreads = 128;


	//setup params
	Cross_accum_params params;
	params.n = batch_size;
	params.c = 1;
	params.h = h;
	params.w = w;
	params.hw = h * w;
	

	params.input_batch_stride = h * w;
	params.output_batch_stride = 3 * h * w;

	params.input = reinterpret_cast<float const *>(x.data_ptr<float>());
	params.output = reinterpret_cast<float *>(output_tensor.data_ptr<float>());
	
	dim3 grid;
	constexpr int32_t elements_perblock = thr_rep * nThreads * 2;
	grid.x = (params.hw + elements_perblock - 1 ) / elements_perblock; grid.y = batch_size; grid.z = 1;
	cross_accum_kernel<thr_rep, elements_perblock><<<grid, nThreads>>>(params);
	return output_tensor;
}