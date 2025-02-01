#include "cten.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor Tensor_new(int numel, TensorShape shape, bool requires_grad) {
    Tensor self;
    memcpy(self.shape, shape, sizeof(TensorShape));
    self.data = malloc(sizeof(FloatBuffer) + sizeof(float) * numel);
    self.data->refcount = 1;
    self.data->numel = numel;
    if(requires_grad) {
        self.node = malloc(sizeof(GradNode));
        memset(self.node, 0, sizeof(GradNode));
    } else {
        self.node = NULL;
    }
    return self;
}

Tensor Tensor_zeros(int numel, TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(numel, shape, requires_grad);
    memset(self.data->flex, 0, sizeof(float) * numel);
    return self;
}

Tensor Tensor_ones(int numel, TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(numel, shape, requires_grad);
    for(int i = 0; i < numel; i++) {
        self.data->flex[i] = 1.0f;
    }
    return self;
}

void Tensor_delete(Tensor self) {
    if(--self.data->refcount == 0) free(self.data);
    if(self.node != NULL) {
        if(self.node->grad.data != NULL) Tensor_delete(self.node->grad);
        free(self.node);
    }
}

Tensor Tensor_detach(Tensor self) {
    Tensor detached = self;
    detached.data->refcount++;
    detached.node = NULL;
    return detached;
}

void Tensor_backward(Tensor self, Tensor grad) {
    if(self.node == NULL) return;
    if(grad.data == NULL) {
        assert(self.data->numel == 1);
        grad = Tensor_ones(1, (TensorShape){0}, false);
    }
    assert(grad.node == NULL);
    if(self.node->grad.data == NULL) {
        self.node->grad = grad;
    } else {
        self.node->grad = Tensor_add(self.node->grad, grad);
    }
    for(int i = 0; i < self.node->n_inputs; i++) {
        grad = Tensor_mul(grad, self.node->grad_fn(self, i));
        Tensor_backward(self.node->inputs[i], grad);
    }
    // Tensor_delete(grad);
}

void Tensor_print(Tensor self) {
    if(self.data == NULL) {
        printf("Tensor()\n");
        return;
    }
    printf("Tensor([");
    for(int i = 0; i < self.data->numel; i++) {
        printf("%.4f", self.data->flex[i]);
        if(i < self.data->numel - 1) printf(", ");
    }
    printf("], shape=(");
    for(int i = 0; i < 4; i++) {
        if(self.shape[i] == 0) {
            break;
        } else {
            if(i > 0) printf(", ");
        }
        printf("%d", self.shape[i]);
    }

    if(self.node != NULL) {
        printf("), grad_fn=<%p>, grad=", self.node->grad_fn);
        Tensor_print(self.node->grad);
    } else {
        printf(")");
    }
    printf(")\n");
}
