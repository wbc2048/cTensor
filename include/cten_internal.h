#pragma once

#include "cten.h"

void* _cten_malloc(size_t size);
void _cten_zero_grad(Tensor* params, int n_params);