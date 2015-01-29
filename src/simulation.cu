#include <cstdio>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "simulation.h"

__global__ void update_units_pos(thrust::pair<float, float> *units_ptr, thrust::pair<float, float> *ends_ptr, unsigned n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    thrust::pair<float, float> new_pos;

    new_pos = ends_ptr[idx];

    __syncthreads();
    units_ptr[idx] = new_pos;
}


void simulation::thread_func() {
    thrust::device_vector<thrust::pair<float, float> > d_units(units.begin(), units.end());
    thrust::device_vector<thrust::pair<float, float> > d_ends;

    while (true) {
        int steps;
        pthread_mutex_lock(&steps_mutex);
        done = true;
        while (steps_shared == 0) {
            pthread_cond_wait(&step_cv, &steps_mutex);
        }
        steps = steps_shared;
        steps_shared = 0;
        done = false;
        pthread_mutex_unlock(&steps_mutex);

        if (steps == -1) {
            break;
        }

        if (updated_ends) {
            d_ends = ends;
        }

        thrust::pair<float, float> *units_ptr = thrust::raw_pointer_cast(&d_units[0]);
        thrust::pair<float, float> *ends_ptr = thrust::raw_pointer_cast(&d_ends[0]);

        unsigned n = units.size();

        const int THREADS_PER_BLOCK = 32;

        dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block(THREADS_PER_BLOCK);

        for (int z = 0; z < steps; ++z) {
            update_units_pos<<<grid, block>>>(units_ptr, ends_ptr, n);
        }

        thrust::copy(d_units.begin(), d_units.end(), units.begin());
    }
    pthread_exit(NULL);
}
