#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstdint>
#include <memory>

namespace {

inline void check_cuda(cudaError_t status, const char* expression) {
  TORCH_CHECK(status == cudaSuccess, expression, " failed: ",
              cudaGetErrorString(status));
}

__global__ void set_conditional_from_mask(cudaGraphConditionalHandle handle,
                                          const int32_t* mask) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphSetConditional(handle,
                            static_cast<unsigned int>(mask[0] != 0));
  }
}

class DeviceConditionalGraph {
 public:
  explicit DeviceConditionalGraph(torch::Tensor device_mask)
      : device_mask_(std::move(device_mask)) {
    TORCH_CHECK(device_mask_.is_cuda(), "device_mask must be CUDA");
    TORCH_CHECK(device_mask_.scalar_type() == torch::kInt32,
                "device_mask must be int32");
    TORCH_CHECK(device_mask_.numel() == 1 && device_mask_.is_contiguous(),
                "device_mask must be one contiguous element");
    device_ = device_mask_.get_device();
    c10::cuda::CUDAGuard guard(device_);

    check_cuda(cudaGraphCreate(&graph_, 0), "cudaGraphCreate");
    check_cuda(cudaGraphConditionalHandleCreate(
                   &conditional_handle_, graph_, 0,
                   cudaGraphCondAssignDefault),
               "cudaGraphConditionalHandleCreate");

    mask_pointer_ = device_mask_.data_ptr<int32_t>();
    setter_arguments_[0] = &conditional_handle_;
    setter_arguments_[1] = &mask_pointer_;
    cudaKernelNodeParams setter_params{};
    setter_params.func =
        reinterpret_cast<void*>(set_conditional_from_mask);
    setter_params.gridDim = dim3(1, 1, 1);
    setter_params.blockDim = dim3(1, 1, 1);
    setter_params.kernelParams = setter_arguments_.data();
    check_cuda(cudaGraphAddKernelNode(&setter_node_, graph_, nullptr, 0,
                                      &setter_params),
               "cudaGraphAddKernelNode(setter)");

    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle = conditional_handle_;
    conditional_params.conditional.type = cudaGraphCondTypeIf;
    conditional_params.conditional.size = 2;
    conditional_params.conditional.phGraph_out = nullptr;
    check_cuda(cudaGraphAddNode(&conditional_node_, graph_, &setter_node_, 1,
                                &conditional_params),
               "cudaGraphAddNode(conditional)");
    branch_graphs_[0] = conditional_params.conditional.phGraph_out[0];
    branch_graphs_[1] = conditional_params.conditional.phGraph_out[1];

    pool_id_ = at::cuda::graph_pool_handle();
  }

  ~DeviceConditionalGraph() {
    if (capturing_) {
      cudaGraph_t ignored = nullptr;
      cudaStreamEndCapture(capture_stream_, &ignored);
      c10::cuda::CUDACachingAllocator::endAllocateToPool(device_, pool_id_);
    }
    if (executable_ != nullptr) {
      cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
    }
    if (pool_registered_) {
      c10::cuda::CUDACachingAllocator::releasePool(device_, pool_id_);
    }
  }

  void capture_begin(int64_t branch_index) {
    TORCH_CHECK(!capturing_, "a branch capture is already active");
    TORCH_CHECK(executable_ == nullptr,
                "cannot capture after graph instantiation");
    TORCH_CHECK(branch_index == 0 || branch_index == 1,
                "branch_index must be 0 (mask true) or 1 (mask false)");
    TORCH_CHECK(!branch_captured_[branch_index],
                "the requested branch was already captured");
    c10::cuda::CUDAGuard guard(device_);
    capture_stream_ = at::cuda::getCurrentCUDAStream(device_).stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    check_cuda(cudaStreamIsCapturing(capture_stream_, &capture_status),
               "cudaStreamIsCapturing");
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "the current stream is already capturing");

    if (!pool_registered_) {
      c10::cuda::CUDACachingAllocator::ensureExistsAndIncrefPool(device_,
                                                                 pool_id_);
      pool_registered_ = true;
    }
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        device_, pool_id_, [stream = capture_stream_](cudaStream_t candidate) {
          return candidate == stream;
        });
    check_cuda(cudaStreamBeginCaptureToGraph(
                   capture_stream_, branch_graphs_[branch_index], nullptr,
                   nullptr, 0, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCaptureToGraph");
    active_branch_ = static_cast<int>(branch_index);
    capturing_ = true;
  }

  void capture_end() {
    TORCH_CHECK(capturing_, "no branch capture is active");
    c10::cuda::CUDAGuard guard(device_);
    cudaGraph_t captured_graph = nullptr;
    cudaError_t status = cudaStreamEndCapture(capture_stream_, &captured_graph);
    c10::cuda::CUDACachingAllocator::endAllocateToPool(device_, pool_id_);
    capturing_ = false;
    check_cuda(status, "cudaStreamEndCapture");
    TORCH_CHECK(captured_graph == branch_graphs_[active_branch_],
                "capture returned an unexpected graph handle");
    branch_captured_[active_branch_] = true;
    active_branch_ = -1;
    capture_stream_ = nullptr;
  }

  void instantiate() {
    TORCH_CHECK(!capturing_, "cannot instantiate during capture");
    TORCH_CHECK(branch_captured_[0] && branch_captured_[1],
                "both branches must be captured before instantiation");
    TORCH_CHECK(executable_ == nullptr, "graph is already instantiated");
    c10::cuda::CUDAGuard guard(device_);
    check_cuda(cudaGraphInstantiate(&executable_, graph_, 0),
               "cudaGraphInstantiate");
  }

  void replay() {
    TORCH_CHECK(executable_ != nullptr, "graph is not instantiated");
    c10::cuda::CUDAGuard guard(device_);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_).stream();
    check_cuda(cudaGraphLaunch(executable_, stream), "cudaGraphLaunch");
  }

 private:
  torch::Tensor device_mask_;
  int device_ = -1;
  int32_t* mask_pointer_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaGraphConditionalHandle conditional_handle_{};
  cudaGraphNode_t setter_node_ = nullptr;
  cudaGraphNode_t conditional_node_ = nullptr;
  std::array<cudaGraph_t, 2> branch_graphs_{nullptr, nullptr};
  std::array<void*, 2> setter_arguments_{nullptr, nullptr};
  std::array<bool, 2> branch_captured_{false, false};
  at::cuda::MempoolId_t pool_id_{};
  cudaStream_t capture_stream_ = nullptr;
  int active_branch_ = -1;
  bool capturing_ = false;
  bool pool_registered_ = false;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<DeviceConditionalGraph,
                  std::shared_ptr<DeviceConditionalGraph>>(
      module, "DeviceConditionalGraph")
      .def(pybind11::init<torch::Tensor>())
      .def("capture_begin", &DeviceConditionalGraph::capture_begin)
      .def("capture_end", &DeviceConditionalGraph::capture_end)
      .def("instantiate", &DeviceConditionalGraph::instantiate)
      .def("replay", &DeviceConditionalGraph::replay);
}
