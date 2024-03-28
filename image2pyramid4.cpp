#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



std::vector<at::Tensor> avgpool_pyramid4_forward_cuda(
	at::Tensor x
);

at::Tensor cross_accum_forward_cuda(
	at::Tensor x
);



std::vector<at::Tensor> avgpool_pyramid4_forward(
	at::Tensor x
){
	CHECK_INPUT(x);
	return avgpool_pyramid4_forward_cuda(x);
}

at::Tensor cross_accum_forward(
	at::Tensor x
){
	CHECK_INPUT(x);
	return cross_accum_forward_cuda(x);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("avgpool_pyramid4_forward", &avgpool_pyramid4_forward, "x");
  m.def("cross_accum_forward", &cross_accum_forward, "x");
}
