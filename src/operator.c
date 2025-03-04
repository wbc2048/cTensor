#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static Tensor GradFn_add(Tensor self, int i) {
    // f(x, y) = x + y; f'(x) = 1; f'(y) = 1
    Tensor input = self.node->inputs[i];
    return Tensor_ones(input.shape, false);
}

static Tensor GradFn_mul(Tensor self, int i) {
    // f(x, y) = x * y; f'(x) = y; f'(y) = x
    return Tensor_detach(self.node->inputs[1 - i]);
}

Tensor Tensor_add(Tensor self, Tensor other) {
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_add() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] + other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_add;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

static Tensor GradFn_sub(Tensor self, int i) {
    // f(x, y) = x - y; f'(x) = 1; f'(y) = -1
    Tensor input = self.node->inputs[i];
    Tensor result = Tensor_ones(input.shape, false);
    if (i == 1) {
        // Negate all elements for the second input
        for(int j = 0; j < result.data->numel; j++) {
            result.data->flex[j] = -result.data->flex[j];
        }
    }
    return result;
}

Tensor Tensor_sub(Tensor self, Tensor other) {
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_sub() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] - other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_sub;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_mul;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

static Tensor GradFn_div(Tensor self, int i) {
    // f(x, y) = x / y
    // f'(x) = 1/y
    // f'(y) = -x/y^2
    Tensor x = Tensor_detach(self.node->inputs[0]);
    Tensor y = Tensor_detach(self.node->inputs[1]);
    
    if (i == 0) {
        // Gradient with respect to x is 1/y
        Tensor result = Tensor_new(x.shape, false);
        for(int j = 0; j < result.data->numel; j++) {
            result.data->flex[j] = 1.0f / y.data->flex[j];
        }
        return result;
    } else {
        // Gradient with respect to y is -x/y^2
        Tensor result = Tensor_new(y.shape, false);
        for(int j = 0; j < result.data->numel; j++) {
            result.data->flex[j] = -x.data->flex[j] / (y.data->flex[j] * y.data->flex[j]);
        }
        return result;
    }
}

Tensor Tensor_div(Tensor self, Tensor other) {
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_div() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] / other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_div;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

static Tensor GradFn_neg(Tensor self, int i) {
    // f(x) = -x; f'(x) = -1
    Tensor input = self.node->inputs[i];
    Tensor result = Tensor_ones(input.shape, false);
    for(int j = 0; j < result.data->numel; j++) {
        result.data->flex[j] = -result.data->flex[j];
    }
    return result;
}

Tensor Tensor_neg(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = -self.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_neg;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

static Tensor GradFn_abs(Tensor self, int i) {
    // f(x) = |x|; f'(x) = x/|x| (sign of x)
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        float x = input.data->flex[j];
        res.data->flex[j] = (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f);
    }
    return res;
}

Tensor Tensor_abs(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = fabsf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_abs;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

static Tensor GradFn_pow(Tensor self, int i) {
    // f(x, y) = x^y
    // f'(x) = y * x^(y-1)
    // f'(y) = x^y * ln(x)
    Tensor x = Tensor_detach(self.node->inputs[0]);
    Tensor y = Tensor_detach(self.node->inputs[1]);
    
    if (i == 0) {
        // Gradient with respect to x is y * x^(y-1)
        Tensor result = Tensor_new(x.shape, false);
        for(int j = 0; j < result.data->numel; j++) {
            float x_val = x.data->flex[j];
            float y_val = y.data->flex[j];
            if (x_val == 0 && y_val <= 0) {
                // Handle the case where x^(y-1) is undefined
                result.data->flex[j] = 0.0f;
            } else {
                result.data->flex[j] = y_val * powf(x_val, y_val - 1.0f);
            }
        }
        return result;
    } else {
        // Gradient with respect to y is x^y * ln(x)
        Tensor result = Tensor_new(y.shape, false);
        for(int j = 0; j < result.data->numel; j++) {
            float x_val = x.data->flex[j];
            float y_val = y.data->flex[j];
            if (x_val <= 0) {
                // Handle the case where ln(x) is undefined
                result.data->flex[j] = 0.0f;
            } else {
                result.data->flex[j] = powf(x_val, y_val) * logf(x_val);
            }
        }
        return result;
    }
}

Tensor Tensor_pow(Tensor self, Tensor other) {
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_pow() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = powf(self.data->flex[i], other.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_pow;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

static Tensor GradFn_min(Tensor self, int i) {
    // f(x) = min(x); f'(x) = 1 at the minimum element, 0 elsewhere
    Tensor input = self.node->inputs[i];
    int min_idx = 0;
    float min_val = input.data->flex[0];
    
    for(int j = 1; j < input.data->numel; j++) {
        if (input.data->flex[j] < min_val) {
            min_val = input.data->flex[j];
            min_idx = j;
        }
    }
    
    Tensor result = Tensor_zeros(input.shape, false);
    result.data->flex[min_idx] = 1.0f;
    return result;
}

Tensor Tensor_min(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){0}, requires_grad);
    
    if (self.data->numel == 0) {
        res.data->flex[0] = 0.0f;
        return res;
    }
    
    float min_val = self.data->flex[0];
    for(int i = 1; i < self.data->numel; i++) {
        if (self.data->flex[i] < min_val) {
            min_val = self.data->flex[i];
        }
    }
    
    res.data->flex[0] = min_val;
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_min;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    
    return res;
}

static Tensor GradFn_max(Tensor self, int i) {
    // f(x) = max(x); f'(x) = 1 at the maximum element, 0 elsewhere
    Tensor input = self.node->inputs[i];
    int max_idx = 0;
    float max_val = input.data->flex[0];
    
    for(int j = 1; j < input.data->numel; j++) {
        if (input.data->flex[j] > max_val) {
            max_val = input.data->flex[j];
            max_idx = j;
        }
    }
    
    Tensor result = Tensor_zeros(input.shape, false);
    result.data->flex[max_idx] = 1.0f;
    return result;
}

Tensor Tensor_max(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){0}, requires_grad);
    
    if (self.data->numel == 0) {
        res.data->flex[0] = 0.0f;
        return res;
    }
    
    float max_val = self.data->flex[0];
    for(int i = 1; i < self.data->numel; i++) {
        if (self.data->flex[i] > max_val) {
            max_val = self.data->flex[i];
        }
    }
    
    res.data->flex[0] = max_val;
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_max;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    
    return res;
}

Tensor Tensor_addf(Tensor self, float other) {
    Tensor tmp = Tensor_new(self.shape, false);
    for(int i = 0; i < tmp.data->numel; i++) {
        tmp.data->flex[i] = other;
    }
    Tensor res = Tensor_add(self, tmp);
    return res;
}

Tensor Tensor_subf(Tensor self, float other) {
    Tensor tmp = Tensor_new(self.shape, false);
    for(int i = 0; i < tmp.data->numel; i++) {
        tmp.data->flex[i] = other;
    }
    Tensor res = Tensor_sub(self, tmp);
    return res;
}

Tensor Tensor_mulf(Tensor self, float other) {
    Tensor tmp = Tensor_new(self.shape, false);
    for(int i = 0; i < tmp.data->numel; i++) {
        tmp.data->flex[i] = other;
    }
    Tensor res = Tensor_mul(self, tmp);
    return res;
}

void Tensor_argmax(Tensor self, int* out) {
    // reduce last dim
    int last_dim = self.shape[TensorShape_dim(self.shape) - 1];
    int n = TensorShape_numel(self.shape) / last_dim;
    for(int i = 0; i < n; i++) {
        float* p = self.data->flex + i * last_dim;
        float max_val = p[0];
        int max_idx = 0;
        for(int j = 1; j < last_dim; j++) {
            if(p[j] > max_val) {
                max_val = p[j];
                max_idx = j;
            }
        }
        out[i] = max_idx;
    }
}

static Tensor GradFn_mean(Tensor self, int i) {
    // f(x) = mean(x); f'(x) = 1 / x.numel()
    Tensor res = Tensor_new(self.shape, false);
    for(int i = 0; i < res.data->numel; i++) {
        res.data->flex[i] = 1.0f / self.data->numel;
    }
    return res;
}

Tensor Tensor_mean(Tensor self) {
    Tensor res = Tensor_new((TensorShape){0}, self.node != NULL);
    float sum = 0;
    for(int i = 0; i < self.data->numel; i++) {
        sum += self.data->flex[i];
    }
    res.data->flex[0] = sum / self.data->numel;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_mean;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

static Tensor GradFn_sum(Tensor self, int i) {
    // f(x) = sum(x); f'(x) = 1
    return Tensor_ones(self.node->inputs[i].shape, false);
}

Tensor Tensor_sum(Tensor self) {
    Tensor res = Tensor_new((TensorShape){0}, self.node != NULL);
    float sum = 0;
    for(int i = 0; i < self.data->numel; i++) {
        sum += self.data->flex[i];
    }
    res.data->flex[0] = sum;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_sum;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

static Tensor GradFn_matmul(Tensor self, int i) {
    Tensor _0 = self.node->inputs[i];
    Tensor _1 = self.node->inputs[1 - i];
    _0 = Tensor_detach(_0);
    _1 = Tensor_detach(_1);
    return Tensor_matmul(_0, _1);
}

Tensor Tensor_matmul(Tensor self, Tensor other) {
    int self_dim = TensorShape_dim(self.shape);
    int other_dim = TensorShape_dim(other.shape);
    assert(self_dim >= 2);
    assert(other_dim >= 2);

    int m = self.shape[self_dim - 2];
    int n = self.shape[self_dim - 1];
    int p = other.shape[other_dim - 1];

    assert(n == other.shape[other_dim - 2]);

    TensorShape res_shape;
    memcpy(res_shape, self.shape, sizeof(TensorShape));
    res_shape[self_dim - 1] = p;
    Tensor res = Tensor_new(res_shape, self.node != NULL || other.node != NULL);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            float sum = 0;
            for(int k = 0; k < n; k++) {
                sum += self.data->flex[i * n + k] * other.data->flex[k * p + j];
            }
            res.data->flex[i * p + j] = sum;
        }
    }
    
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_matmul;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    
    return res;
}