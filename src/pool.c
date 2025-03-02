#include "cten.h"

#include "common/vector.h"
#include <stddef.h>

typedef struct {
    c11_vector /*PoolId*/ stack;
    c11_vector /*void_p*/ pointers;
    c11_vector /*void_p*/ pointers_swap_buffer;
} PoolAllocator;

static PoolAllocator g_allocator;

void cten_initilize() {
    c11_vector__ctor(&g_allocator.stack, sizeof(PoolId));
    c11_vector__ctor(&g_allocator.pointers, sizeof(void*));
    c11_vector__ctor(&g_allocator.pointers_swap_buffer, sizeof(void*));
}

void cten_finalize() {
    for(int i = 0; i < g_allocator.pointers.length; i++) {
        void* p = c11__getitem(void*, &g_allocator.pointers, i);
        free(p);
    }
    assert(g_allocator.pointers_swap_buffer.length == 0);
    c11_vector__dtor(&g_allocator.stack);
    c11_vector__dtor(&g_allocator.pointers);
    c11_vector__dtor(&g_allocator.pointers_swap_buffer);
}

void cten_begin_malloc(PoolId id) {
    c11_vector* self = &g_allocator.stack;
    c11_vector__push(PoolId, self, id);
}

void cten_end_malloc() {
    c11_vector* self = &g_allocator.stack;
    assert(self->length > 0);
    c11_vector__pop(self);
}

void cten_free(PoolId id) {
    c11_vector* pointers = &g_allocator.pointers;
    c11_vector* swap_buffer = &g_allocator.pointers_swap_buffer;
    for(int i = 0; i < pointers->length; i++) {
        void* p = c11__getitem(void*, pointers, i);
        if(((PoolId*)p)[0] == id) {
            free(p);
        } else {
            c11_vector__push(void*, swap_buffer, p);
        }
    }
    c11_vector__swap(pointers, swap_buffer);
    c11_vector__clear(swap_buffer);
}

void* _cten_malloc(size_t size) {
    assert(g_allocator.stack.length > 0);
    PoolId id = c11_vector__back(PoolId, &g_allocator.stack);
    c11_vector* pointers = &g_allocator.pointers;
    void* p = malloc(sizeof(PoolId) + size);
    assert(p != NULL);
    ((PoolId*)p)[0] = id;
    c11_vector__push(void*, pointers, p);
    return (PoolId*)p + 1;
}
