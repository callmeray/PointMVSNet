#include "gather_knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_knn_forward", &GatherKNNForward, "Gather KNN forward (CUDA)");
  m.def("gather_knn_backward", &GatherKNNBackward, "Gather KNN backward (CUDA)");
}
