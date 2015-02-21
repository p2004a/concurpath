#include <cstdio>
#include <cassert>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include "simulation.h"

__device__ __host__
int get_sector(int x, int y, int width) {
    return  (y / MAP_SECTOR_SIZE) * ((width + MAP_SECTOR_SIZE - 1) / MAP_SECTOR_SIZE) + (x / MAP_SECTOR_SIZE);
}

__global__ void update_units_pos(
    thrust::pair<float, float> *units_ptr,
    thrust::pair<float, float> *ends_ptr,
    unsigned n,
    bool *map,
    int width,
    int height,
    thrust::pair<int, int> *sectors_map,
    int *indexes
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
                int pos_x = floor(pos.first + dx);
                int pos_y = floor(pos.second + dy);
                if (pos_x >= 0 && pos_x < width && pos_y >= 0 && pos_y < height) {
                    wall[dy + 1][dx + 1] = map[pos_y * width + pos_x];
                }
            }
        }
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (wall[dy + 1][dx + 1]) {
                    float wall_dx = 0.0;
                    float wall_dy = 0.0;
                    if (dx) {
                        float wall_center_x = floor(pos.first + dx) + 0.5;
                        wall_dx = pos.first - (wall_center_x - dx * 0.5);
                    }
                    if (dy) {
                        float wall_center_y = floor(pos.second + dy) + 0.5;
                        wall_dy = pos.second - (wall_center_y - dy * 0.5);
                    }
                    if (dx && dy) {
                        if (!wall[1][dx+1] && !wall[dy+1][1]) {
                            float d = hypot(wall_dx, wall_dy);
                            float force = 1 / (d * d) - 1.0;
                            if (force > 0.0) {
                                f.first += (wall_dx / d) * force;
                                f.second += (wall_dy / d) * force;
                            }
                        }
                    } else if (dx) {
                        f.first += copysign(1 / (wall_dx * wall_dx) - 1.0, wall_dx);
                    } else if (dy) {
                        f.second += copysign(1 / (wall_dy * wall_dy) - 1.0, wall_dy);
                    }
                }
            }
        }
    }

    // force from other units
    {
        int sector_x = (int)pos.first / MAP_SECTOR_SIZE;
        int sector_y = (int)pos.second / MAP_SECTOR_SIZE;
        int max_sector_width = (width + 3) / MAP_SECTOR_SIZE;
        int max_sector_height = (height + 3) / MAP_SECTOR_SIZE;
        thrust::pair<int, int> sectors[3][3];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                sectors[dy + 1][dx + 1].first = -1;
                int pos_x = sector_x + dx;
                int pos_y = sector_y + dy;
                if (pos_x >= 0 && pos_x < max_sector_width && pos_y >= 0 && pos_y < max_sector_height) {
                    sectors[dy + 1][dx + 1] = sectors_map[pos_y * max_sector_width + pos_x];
                }
            }
        }
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (sectors[dy + 1][dx + 1].first != -1) {
                    for (int i = 0; i < sectors[dy + 1][dx + 1].second; ++i) {
                        int j = sectors[dy + 1][dx + 1].first + i;
                        if (j != idx) {
                            float x = units_ptr[j].first - pos.first;
                            float y = units_ptr[j].second - pos.second;
                            float d_reciprocal = rsqrt(x * x + y * y);
                            float force = d_reciprocal + (0.5 * d_reciprocal * d_reciprocal) - 0.285;
                            if (force > 0) {
                                f.first += -x * d_reciprocal * force;
                                f.second += -y * d_reciprocal * force;
                            }
                        }
                    }
                }
            }
        }
    }

    // displacement
    {
        float vec_d_reciprocal = rsqrt(f.first * f.first + f.second * f.second);
        const float absolute_displacement = 0.007;
        new_pos.first = pos.first + f.first * vec_d_reciprocal * absolute_displacement;
        new_pos.second = pos.second + f.second * vec_d_reciprocal * absolute_displacement;
    }

    __syncthreads();
    units_ptr[idx] = new_pos;
}

class sectors_functor {
    const int width;
  public:
    sectors_functor(int _width) : width(_width) {}

    __host__ __device__
    int operator()(thrust::pair<float, float> const& pos) const {
        return get_sector(pos.first, pos.second, width);
    }
};

__global__
void update_sectors_map(
    int *sectors_indexes,
    int *sectors,
    int n,
    thrust::pair<int, int> *sectors_map
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }
    thrust::pair<int, int> & map_field = sectors_map[sectors[idx]];
    map_field.first = sectors_indexes[idx];
    map_field.second = sectors_indexes[idx + 1] - map_field.first;
}

void simulation::thread_func() {
    const unsigned n = units.size();
    const int THREADS_PER_BLOCK = 128;

    thrust::device_vector<thrust::pair<float, float> > d_units(units.begin(), units.end());
    thrust::device_vector<thrust::pair<float, float> > d_ends;
    thrust::device_vector<thrust::pair<float, float> > d_units_copy(units.size());
    thrust::device_vector<thrust::pair<float, float> > d_ends_copy(units.size());
    thrust::device_vector<thrust::pair<int, int> > d_sectors_map(sectors_map.size());
    thrust::device_vector<int> sectors(n), units_indexes(n), sectors_indexes(n+1);
    thrust::device_vector<bool> d_map(map.begin(), map.end());

    pthread_mutex_lock(&steps_mutex);
    pthread_cond_signal(&step_cv);
    pthread_mutex_unlock(&steps_mutex);

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

        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        for (int z = 0; z < steps; ++z) {
            thrust::transform(d_units.begin(), d_units.end(), sectors.begin(), sectors_functor(map_width));
            thrust::sequence(units_indexes.begin(), units_indexes.end());
            thrust::stable_sort_by_key(sectors.begin(), sectors.end(), units_indexes.begin());
            thrust::sequence(sectors_indexes.begin(), sectors_indexes.end());
            thrust::device_vector<int>::iterator sectors_new_end = thrust::unique_by_key(sectors.begin(), sectors.end(), sectors_indexes.begin()).first;
            int after_unique_sectors = sectors_new_end - sectors.begin();
            sectors_indexes[after_unique_sectors] = n; // guard
            thrust::fill(d_sectors_map.begin(), d_sectors_map.end(), thrust::pair<int, int>(-1, 0));

            int *sectors_indexes_ptr = thrust::raw_pointer_cast(&sectors_indexes[0]);
            int *sectors_ptr = thrust::raw_pointer_cast(&sectors[0]);
            thrust::pair<int, int> *sectors_map_ptr = thrust::raw_pointer_cast(&d_sectors_map[0]);

            dim3 grid_sectors((after_unique_sectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            dim3 block_sectors(THREADS_PER_BLOCK);

            update_sectors_map<<<grid_sectors, block_sectors>>>(
                sectors_indexes_ptr, sectors_ptr, after_unique_sectors, sectors_map_ptr
            );

            int *units_indexes_ptr = thrust::raw_pointer_cast(&units_indexes[0]);

            thrust::gather(units_indexes.begin(), units_indexes.end(), d_units.begin(), d_units_copy.begin());
            thrust::gather(units_indexes.begin(), units_indexes.end(), d_ends.begin(), d_ends_copy.begin());

            thrust::pair<float, float> *units_ptr = thrust::raw_pointer_cast(&d_units_copy[0]);
            thrust::pair<float, float> *ends_ptr = thrust::raw_pointer_cast(&d_ends_copy[0]);
            bool *map_ptr = thrust::raw_pointer_cast(&d_map[0]);

            dim3 grid_units((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            dim3 block_units(THREADS_PER_BLOCK);

            update_units_pos<<<grid_units, block_units>>>(
                units_ptr, ends_ptr, n, map_ptr, map_width, map_height, sectors_map_ptr, units_indexes_ptr);

            thrust::scatter(d_units_copy.begin(), d_units_copy.end(), units_indexes.begin(), d_units.begin());
        }

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &t2);
        unsigned long long diff = (long long)(t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
        kernel_time = diff / steps;

        thrust::copy(d_units.begin(), d_units.end(), units.begin());
        thrust::copy(d_sectors_map.begin(), d_sectors_map.end(), sectors_map.begin());
    }
    pthread_exit(NULL);
}
