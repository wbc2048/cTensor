#include "cten.h"
#include "cten_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct optim_sgd {
    int n_params;
    Tensor* params;
    
    float lr;
    float momentum;
    float weight_decay;

    Tensor* velocity;
} optim_sgd;

optim_sgd* optim_sgd_new(int n_params, Tensor* params) {
    optim_sgd* self = malloc(sizeof(optim_sgd));
    self->n_params = n_params;
    self->params = params;
    return self;
}

void optim_sgd_config(optim_sgd* self, float lr, float momentum, float weight_decay) {
    self->lr = lr;
    self->momentum = momentum;
    self->weight_decay = weight_decay;
}

static void optim_sgd_update_f(Tensor t, void* ctx){
    optim_sgd* self = (optim_sgd*)ctx;
}

void optim_sgd_update(optim_sgd* self, Tensor graph) {
    int n_params = Tensor_backward_apply(graph, NULL, NULL);
    self->n_params = n_params;
    self->velocity = malloc(sizeof(Tensor) * n_params);
    Tensor_backward_apply(graph, optim_sgd_update_f, self);
}