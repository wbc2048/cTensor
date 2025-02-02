#include "cten.h"

typedef struct Model {
    Tensor weight_1, weight_2;
    Tensor bias_1, bias_2;
} Model;

Tensor model_forward(Model* model, Tensor input) {
    // input: [None, 4]
    Tensor x = nn_linear(input, model->weight_1, model->bias_1);
    x = nn_relu(x);
    x = nn_linear(x, model->weight_2, model->bias_2);
    x = nn_softmax(x, 1);
    return x;
}

int main() {
    // Tensor a = Tensor_ones(4, (TensorShape){2, 2}, true);
    // Tensor b = Tensor_mulf(a, 2.0f);

    // Tensor_print(a);
    // Tensor_print(b);

    // Tensor_backward(b, Tensor_ones(4, (TensorShape){2, 2}, false));

    // Tensor_print(a);
    // Tensor_print(b);

    // load iris dataset
    const float (*X)[4];
    const int* y;
    int n_samples = load_iris_dataset(&X, &y);

    // create model
    Model model;
    model.weight_1 = Tensor_new((TensorShape){4, 32}, true);
    model.bias_1 = Tensor_new((TensorShape){32}, true);
    model.weight_2 = Tensor_new((TensorShape){32, 3}, true);
    model.bias_2 = Tensor_new((TensorShape){3}, true);

    // train model
    optim_sgd* optimizer = optim_sgd_new(0.001, 0, 0);

    int batch_size = 8;
    for(int epoch = 0; epoch < 5; epoch++) {
        for(int i = 0; i < n_samples; i += batch_size) {
            // create input and target tensors
            Tensor input = Tensor_new((TensorShape){batch_size, 4}, false);
            Tensor y_true = Tensor_new((TensorShape){batch_size}, false);
            for(int j = 0; j < batch_size; j++) {
                for(int k = 0; k < 4; k++) {
                    input.data->flex[j * 4 + k] = X[i + j][k];
                }
                y_true.data->flex[j] = y[i + j];
            }
            // forward pass
            Tensor y_pred = model_forward(&model, input);
            // compute loss
            Tensor loss = nn_crossentropy(y_true, y_pred);
            // backward pass
            Tensor_backward(loss, (Tensor){});
            // update weights
            optim_sgd_update(optimizer, loss.node);
        }
    }
    return 0;
}

// b = 2 * a
// b' = 2