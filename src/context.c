#include "cten.h"
#include "cten_internal.h"

static int _eval_depth = 0;

void cten_begin_eval() { _eval_depth++; }

bool cten_is_eval() { return _eval_depth > 0; }

void cten_end_eval() { _eval_depth--; }