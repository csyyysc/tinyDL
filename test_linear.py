import numpy as np
import time
from linear_layer import Linear  # 自定義 CUDA 線性層

# 測試配置
batch_size = 8192
in_features = 4096
out_features = 2048


# 隨機生成數據
x = np.random.randn(batch_size, in_features).astype(np.float32)
grad_output = np.random.randn(batch_size, out_features).astype(np.float32)

# 初始化線性層
linear = Linear(in_features, out_features)

# ---- CUDA 測試 ----
start = time.time()
cuda_output = linear.forward(x)
cuda_forward_time = time.time() - start

start = time.time()
cuda_grad_input = linear.backward(grad_output, learning_rate=0.01)
cuda_backward_time = time.time() - start

# ---- NumPy 測試 ----
weights = linear.weights.copy()
bias = linear.bias.copy()

start = time.time()
numpy_output = np.dot(x, weights) + bias
numpy_forward_time = time.time() - start

start = time.time()
numpy_grad_weights = np.dot(x.T, grad_output)
numpy_grad_bias = np.sum(grad_output, axis=0, keepdims=True)
numpy_grad_input = np.dot(grad_output, weights.T)
numpy_backward_time = time.time() - start

# ---- 結果比較 ----
print("=== Performance Comparison ===")
print(f"CUDA Forward Time: {cuda_forward_time:.6f} seconds")
print(f"NumPy Forward Time: {numpy_forward_time:.6f} seconds")
print(f"CUDA Backward Time: {cuda_backward_time:.6f} seconds")
print(f"NumPy Backward Time: {numpy_backward_time:.6f} seconds")
