# import torch before loading cuda extension
import torch

try:
    from pointmvsnet.functions import dgcnn_ext
except ImportError:
    print("Please compile source files before using dgcnn cuda extension.")


class GatherKNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, index):
        ctx.save_for_backward(index)
        feature_neighbour = dgcnn_ext.gather_knn_forward(feature, index)
        return feature_neighbour

    @staticmethod
    def backward(ctx, grad_output):
        knn_inds = ctx.saved_tensors[0]
        grad_features = dgcnn_ext.gather_knn_backward(grad_output, knn_inds)
        return grad_features, None


gather_knn = GatherKNN.apply


def test_gather_knn():
    torch.manual_seed(1)
    batch_size = 2
    num_inst = 5
    channels = 4
    k = 3

    feature_tensor = torch.rand(batch_size, channels, num_inst).cuda(0)
    # knn_inds = torch.ones([batch_size, num_inst, k], dtype=torch.int64).cuda(0)
    # knn_inds[:, :, 2] = 2
    # knn_inds[:, 0, 2] = 3
    knn_inds = torch.randint(0, num_inst, [batch_size, num_inst, k]).long().cuda(0)

    feature_tensor_gather = torch.zeros_like(feature_tensor).copy_(feature_tensor)
    feature_tensor_gather.requires_grad = True
    feature_tensor_cuda = torch.zeros_like(feature_tensor).copy_(feature_tensor)
    feature_tensor_cuda.requires_grad = True

    feature_expand = feature_tensor_gather.unsqueeze(2).expand(batch_size, channels, num_inst, num_inst)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_inst, k)
    feature_gather = torch.gather(feature_expand, 3, knn_inds_expand)

    feature_cuda = gather_knn(feature_tensor_cuda, knn_inds)
    print("Forward:", feature_gather.allclose(feature_cuda))

    feature_gather.backward(torch.ones_like(feature_gather))
    feature_cuda.backward(torch.ones_like(feature_cuda))
    grad_gather = feature_tensor_gather.grad
    grad_cuda = feature_tensor_cuda.grad
    print("Backward:", grad_gather.allclose(grad_cuda))


if __name__ == "__main__":
    test_gather_knn()
