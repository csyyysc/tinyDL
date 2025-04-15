#include "Tensor.cuh"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

// void test_add() {
//     auto x = std::make_shared<Tensor>(1, 1, true);
//     auto y = std::make_shared<Tensor>(1, 1, true);

//     x->data[0] = 2.0f;
//     y->data[0] = 3.0f;

//     auto z = add(x, y); // z = x + y
//     z->backward();      // dz/dx = 1, dz/dy = 1

//     std::cout << "x.grad = " << x->grad[0] << std::endl;
//     std::cout << "y.grad = " << y->grad[0] << std::endl;
// }

void test_init_default_tensor() {
    std::vector<int> shape = {2, 3, 4};
    Tensor tensor(shape);

    assert(tensor.size() == 2 * 3 * 4);
    assert(tensor.device_id == 0);
    assert(tensor.shape_vec == shape);
    assert(tensor.requires_grad == false);
    assert(tensor.data != nullptr);
    assert(tensor.grad == nullptr);

    assert(tensor.strides.size() == shape.size());
    assert(tensor.strides[shape.size() - 1] == 1);
    for (int i = 0; i <= shape.size() - 2; ++i) {
        assert(tensor.strides[i] == tensor.strides[i + 1] * shape[i + 1]);
    }
}

void test_init_custom_tensor() {
    std::vector<int> shape = {28, 28, 10};
    Tensor tensor(shape, true, 1);

    assert(tensor.size() == 28 * 28 * 10);
    assert(tensor.device_id == 1);
    assert(tensor.shape_vec == shape);
    assert(tensor.requires_grad == true);
    assert(tensor.data != nullptr);
    assert(tensor.grad != nullptr);

    assert(tensor.strides.size() == shape.size());
    assert(tensor.strides[shape.size() - 1] == 1);
    for (int i = 0; i <= shape.size() - 2; ++i) {
        assert(tensor.strides[i] == tensor.strides[i + 1] * shape[i + 1]);
    }
}

int main() {
    test_init_default_tensor();
    test_init_custom_tensor();

    return 0;
}
