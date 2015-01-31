#include <cstdio>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "simulation.h"

__global__ void update_units_pos(
    thrust::pair<float, float> *units_ptr,
    thrust::pair<float, float> *ends_ptr,
    unsigned n,
    bool *map,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    thrust::pair<float, float> new_pos, pos, f;

    pos = units_ptr[idx];

    // force to end
    {
        const float end_force = 4.0;
        float x = ends_ptr[idx].first - pos.first;
        float y = ends_ptr[idx].second - pos.second;
        float end_d_reciprocal = rsqrt(x * x + y * y);
        f.first = x * end_d_reciprocal * end_force;
        f.second = y * end_d_reciprocal * end_force;
    }

    // force from walls
    {
        bool wall[3][3];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                wall[dy + 1][dx + 1] = true;
                int pos_x = pos.first + dx;
                int pos_y = pos.second + dy;
                if (pos_x >= 0 && pos_x < width && pos_y >= 0 && pos_y < height) {
                    wall[dy + 1][dx + 1] = map[pos_y * width + pos_x];
                }
            }
        }
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (wall[dy + 1][dx + 1]) {
                    float force_x = 0.0;
                    float force_y = 0.0;
                    float wall_center_x = floor(pos.first + dx) + 0.5;
                    float wall_center_y = floor(pos.second + dy) + 0.5;
                    if (dx && dy) {
                        if (!wall[1][dx+1] && !wall[dy+1][1]) {
                            float x = (wall_center_x - dx * 0.5) - pos.first;
                            float y = (wall_center_y - dy * 0.5) - pos.second;
                            float d = hypot(x, y);
                            float force = 1 / (d * d) - 1.0;
                            force_x = (-x / d) * force;
                            force_y = (-y / d) * force;
                        }
                    } else if (dx) {
                        float wall_dx = pos.first - (wall_center_x - dx * 0.5);
                        force_x = copysign(1 / (wall_dx * wall_dx) - 1.0, wall_dx);
                    } else if (dy) {
                        float wall_dy = pos.second - (wall_center_y - dy * 0.5);
                        force_y = copysign(1 / (wall_dy * wall_dy) - 1.0, wall_dy);
                    }
                    f.first += force_x;
                    f.second += force_y;
                }
            }
        }
    }

    // force from other units
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

    // displacement
    {
        float vec_d_reciprocal = rsqrt(f.first * f.first + f.second * f.second);
        const float absolute_displacement = 0.001;
        new_pos.first = pos.first + f.first * vec_d_reciprocal * absolute_displacement;
        new_pos.second = pos.second + f.second * vec_d_reciprocal * absolute_displacement;
    }

    __syncthreads();
    units_ptr[idx] = new_pos;
}


void simulation::thread_func() {
    thrust::device_vector<thrust::pair<float, float> > d_units(units.begin(), units.end());
    thrust::device_vector<thrust::pair<float, float> > d_ends;
    thrust::device_vector<bool> d_map(map.begin(), map.end());

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
        bool *map_ptr = thrust::raw_pointer_cast(&d_map[0]);

        unsigned n = units.size();

        const int THREADS_PER_BLOCK = 256;

        dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block(THREADS_PER_BLOCK);

        for (int z = 0; z < steps; ++z) {
            update_units_pos<<<grid, block>>>(
                units_ptr, ends_ptr, n, map_ptr, map_width, map_height);
        }

        thrust::copy(d_units.begin(), d_units.end(), units.begin());
    }
    pthread_exit(NULL);
}
