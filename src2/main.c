#include "cten.h"

int main() {
    Tensor a = Tensor_ones(4, (TensorShape){2, 2}, true);
    Tensor b = Tensor_mulf(a, 2.0f);

    Tensor_print(a);
    Tensor_print(b);

    Tensor_backward(b, Tensor_ones(4, (TensorShape){2, 2}, false));

    Tensor_print(a);
    Tensor_print(b);

    return 0;
}

// b = 2 * a
// b' = 2