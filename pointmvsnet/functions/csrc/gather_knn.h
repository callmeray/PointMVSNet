#ifndef _GATHER_KNN
#define _GATHER_KNN

#include <torch/extension.h>

// CUDA declarations
at::Tensor GatherKNNForward(
    const at::Tensor input,
    const at::Tensor index);

at::Tensor GatherKNNBackward(
    const at::Tensor grad_output,
    const at::Tensor index);

#endif
