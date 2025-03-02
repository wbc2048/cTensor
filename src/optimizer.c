#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct optim_sgd {
    int n_params;
    Tensor* params;
    
    float lr;
    float momentum;

    Tensor* velocity;
} optim_sgd;

optim_sgd* optim_sgd_new(int n_params, Tensor* params) {
    optim_sgd* self = malloc(sizeof(optim_sgd));
    self->n_params = n_params;
    self->params = params;
    self->lr = 0.001f;
    self->momentum = 0.0f;
    return self;
}

void optim_sgd_config(optim_sgd* self, float lr, float momentum) {
    self->lr = lr;
    self->momentum = momentum;
}

void optim_sgd_zerograd(optim_sgd *self){
    for(int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if(t.node != NULL && t.node->grad.data != NULL) {
            t.node->grad = Tensor_zeros(t.shape, false);
        }
    }
}

void optim_sgd_step(optim_sgd* self) {
    for(int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        assert(self->momentum == 0);
        assert(t.node != NULL);
        assert(t.node->grad.data != NULL);
        // step
        for(int j = 0; j < t.data->numel; j++) {
            t.data->flex[j] -= self->lr * t.node->grad.data->flex[j];
        }
    }
}