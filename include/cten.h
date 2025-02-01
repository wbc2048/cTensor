#pragma once

#include <stdbool.h>

typedef int TensorShape[4];
typedef struct GradNode GradNode;

typedef struct FloatBuffer {
    int refcount;
    int numel;
    float flex[];
} FloatBuffer;

typedef struct Tensor {
    TensorShape shape;
    FloatBuffer* data;
    GradNode* node;
} Tensor;

typedef struct GradNode {
    struct Tensor grad;
    struct Tensor (*grad_fn)(struct Tensor self, int i);
    struct Tensor inputs[4];
    int n_inputs;
} GradNode;

Tensor Tensor_new(int numel, TensorShape shape, bool requires_grad);
Tensor Tensor_zeros(int numel, TensorShape shape, bool requires_grad);
Tensor Tensor_ones(int numel, TensorShape shape, bool requires_grad);
void Tensor_delete(Tensor self);

Tensor Tensor_detach(Tensor self);
void Tensor_backward(Tensor self, Tensor grad);

void Tensor_print(Tensor self);

Tensor Tensor_add(Tensor self, Tensor other);
Tensor Tensor_sub(Tensor self, Tensor other);
Tensor Tensor_mul(Tensor self, Tensor other);
Tensor Tensor_div(Tensor self, Tensor other);
Tensor Tensor_pow(Tensor self, Tensor other);

Tensor Tensor_addf(Tensor self, float other);
Tensor Tensor_subf(Tensor self, float other);
Tensor Tensor_mulf(Tensor self, float other);
Tensor Tensor_divf(Tensor self, float other);
Tensor Tensor_powf(Tensor self, float other);

Tensor Tensor_neg(Tensor self);
Tensor Tensor_abs(Tensor self);

Tensor Tensor_sum(Tensor self, int dim);
Tensor Tensor_mean(Tensor self, int dim);
Tensor Tensor_max(Tensor self, int dim);
Tensor Tensor_min(Tensor self, int dim);

Tensor nn_log(Tensor self);
Tensor nn_exp(Tensor self);

Tensor nn_sin(Tensor self);
Tensor nn_cos(Tensor self);
Tensor nn_tan(Tensor self);

Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);
Tensor nn_relu(Tensor input);
Tensor nn_sigmoid(Tensor input);
Tensor nn_tanh(Tensor input);
Tensor nn_softmax(Tensor input, int dim);
