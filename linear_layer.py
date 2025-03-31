import numpy as np
import cuda_nn  # 假設已編譯的 CUDA 模塊名

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 初始化權重和偏置
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * 0.01
        self.bias = np.zeros((1, out_features), dtype=np.float32)

        # 用於保存梯度
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        """
        前向傳播: y = xW + b
        """
        self.input = x
        self.output = np.zeros((x.shape[0], self.out_features), dtype=np.float32)

        # CUDA 加速的矩陣乘法
        cuda_nn.matrix_multiply(x, self.weights, self.output, x.shape[0], self.in_features, self.out_features)

        # 添加偏置
        self.output += self.bias
        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        """
        反向傳播: 計算梯度並更新權重
        """
        # 確保輸入矩陣的順序一致
        input_t = np.ascontiguousarray(self.input.T)  # 確保為 column-major
        grad_output_t = np.ascontiguousarray(grad_output.T)

        # 計算梯度: dL/dW = x^T * dL/dy
        self.grad_weights = np.zeros_like(self.weights)
        cuda_nn.matrix_multiply(input_t, grad_output, self.grad_weights, 
                                self.in_features, self.input.shape[0], self.out_features)

        # 計算輸入梯度: dL/dx = dL/dy * W^T
        grad_input = np.zeros_like(self.input)
        weights_t = np.ascontiguousarray(self.weights.T)  # 確保為 column-major
        cuda_nn.matrix_multiply(grad_output, weights_t, grad_input, 
                                grad_output.shape[0], self.out_features, self.in_features)

        # 計算偏置梯度: dL/db = sum(dL/dy)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # 更新權重和偏置
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

        return grad_input
