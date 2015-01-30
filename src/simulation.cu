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

    thrust::pair<float, float> new_pos, pos, f;

    pos = units_ptr[idx];

    {
        const float end_force = 2.0;
        float x = ends_ptr[idx].first - pos.first;
        float y = ends_ptr[idx].second - pos.second;
        float end_d_reciprocal = rsqrt(x * x + y * y);
        f.first = x * end_d_reciprocal * end_force;
        f.second = y * end_d_reciprocal * end_force;
    }

    for (int i = 0; i < n; ++i) {
        if (i != idx) {
            float x = units_ptr[i].first - pos.first;
            float y = units_ptr[i].second - pos.second;
            float d_reciprocal = rsqrt(x * x + y * y);
            float force = d_reciprocal - 0.3;
            if (force > 0) {
                f.first += -x * d_reciprocal * force;
                f.second += -y * d_reciprocal * force;
            }
        }
    }

    float vec_d_reciprocal = rsqrt(f.first * f.first + f.second * f.second);

    new_pos.first = pos.first + f.first * vec_d_reciprocal * 0.002;
    new_pos.second = pos.second + f.second * vec_d_reciprocal * 0.002;

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
