// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  task-tapir.c
  Task-Tapir interface
*/

#include "julia.h"
#include "julia_internal.h"

static jl_function_t *taskgroup_fun JL_GLOBALLY_ROOTED = NULL;
static jl_function_t *spawn_fun JL_GLOBALLY_ROOTED = NULL;
static jl_function_t *sync_fun JL_GLOBALLY_ROOTED = NULL;

JL_DLLEXPORT jl_value_t *jl_tapir_taskgroup(void)
{
    // Initialization pattern from `task_done_hook_func`:
    jl_function_t *impl = jl_atomic_load_relaxed(&taskgroup_fun);
    if (!impl) {
        impl = jl_get_function(jl_base_module, "_Tapir_taskgroup");
        jl_atomic_store_release(&taskgroup_fun, impl);
    }
    assert(impl);
    jl_value_t *ans;
    jl_value_t *argv[1];
    argv[0] = (jl_value_t*)impl;
    ans = jl_apply(argv, 1);
    return ans;
}

JL_DLLEXPORT void jl_tapir_spawn(jl_value_t *taskgroup, void *f, uint8_t *arg,
                                 size_t arg_size, jl_value_t **jvs, size_t jvs_size,
                                 size_t world_age)
{
    jl_function_t *impl = jl_atomic_load_relaxed(&spawn_fun);
    if (!impl) {
        impl = jl_get_function(jl_base_module, "_Tapir_spawn!");
        jl_atomic_store_release(&spawn_fun, impl);
    }
    assert(impl);
    jl_value_t **argv;
    jl_array_t *root = jl_alloc_array_1d(jl_array_any_type, jvs_size);
    for (size_t i = 0; i < jvs_size; i++) {
        jl_array_ptr_set(root, i, jvs[i]);
    }
    JL_GC_PUSHARGS(argv, 7);
    argv[5] = (jl_value_t*)root;
    argv[0] = (jl_value_t*)impl;
    argv[1] = taskgroup;
    argv[2] = jl_box_voidpointer(f);
    argv[3] = jl_box_uint8pointer(arg);
    argv[4] = jl_box_ulong(arg_size); // TODO: jl_box_csize_t?
    argv[6] = jl_box_ulong(world_age);
    jl_apply(argv, 7);
    JL_GC_POP();
    // Not using `jl_call` to propagate exception.
}

JL_DLLEXPORT void jl_tapir_sync(jl_value_t *taskgroup)
{
    jl_function_t *impl = jl_atomic_load_relaxed(&sync_fun);
    if (!impl) {
        impl = jl_get_function(jl_base_module, "_Tapir_sync!");
        jl_atomic_store_release(&sync_fun, impl);
    }
    assert(impl);
    jl_value_t **argv;
    JL_GC_PUSHARGS(argv, 2);
    argv[0] = (jl_value_t*)impl;
    argv[1] = taskgroup;
    jl_apply(argv, 2);
    JL_GC_POP();
}
