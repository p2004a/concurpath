#pragma once

/*
    Unfortunatelly I have to use pthreads not c++11 threads because
    nvcc doesn't support C++11 yet and I don't wan't to contrive
*/

#include <pthread.h>

#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAP_SECTOR_SIZE 4

class simulation {
    thrust::host_vector<thrust::pair<float, float> > units;
    thrust::host_vector<thrust::pair<float, float> > ends;
    thrust::host_vector<thrust::pair<int, int> > sectors_map;
    thrust::host_vector<bool> map;
    int map_width, map_height, sectors_map_width, sectors_map_height;

    pthread_t thread;
    pthread_mutex_t steps_mutex;
    pthread_cond_t step_cv;
    unsigned long long kernel_time;
    volatile bool done;
    volatile int steps_shared;

    volatile bool updated_ends;

    typedef thrust::host_vector<thrust::pair<float, float> >::const_iterator u_const_iterator;
    typedef thrust::host_vector<thrust::pair<int, int> >::const_iterator sm_const_iterator;

    void thread_func();
    static void* thread_func_helper(void *context) {
        ((simulation *)context)->thread_func();
        return NULL;
    }

    simulation(simulation const &) {};
  public:

    template<typename It, typename Arr2D>
    simulation(It begin, It end, Arr2D const& arr, int w, int h)
      : units(begin, end), map(w * h), map_width(w), map_height(h), steps_shared(0), updated_ends(true) {

        ends.resize(units.size());
        for (unsigned i = 0; i < units.size(); ++i) {
            ends[i].first = -1;
        }

        #pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                map[y * w + x] = arr[y][x];
            }
        }

        sectors_map_width = (map_width + MAP_SECTOR_SIZE - 1) / MAP_SECTOR_SIZE;
        sectors_map_height = (map_height + MAP_SECTOR_SIZE - 1) / MAP_SECTOR_SIZE;
        sectors_map.resize(sectors_map_width * sectors_map_height);

        pthread_mutex_init(&steps_mutex, NULL);
        pthread_cond_init(&step_cv, NULL);

        pthread_mutex_lock(&steps_mutex);

        int res = pthread_create(&thread, NULL, &simulation::thread_func_helper, (void *)this);
        if (res) {
            pthread_mutex_unlock(&steps_mutex);
            throw std::runtime_error("failed to create thread");
        }

        pthread_cond_wait(&step_cv, &steps_mutex);
        pthread_mutex_unlock(&steps_mutex);
    }

    ~simulation() {
        run(-1);

        int res = pthread_join(thread, NULL);
        if (res) {
            throw std::runtime_error("failed to join thread");
        }

        pthread_mutex_destroy(&steps_mutex);
        pthread_cond_destroy(&step_cv);
    }

    u_const_iterator u_begin() const {
        return units.begin();
    }

    u_const_iterator u_end() const {
        return units.end();
    }

    sm_const_iterator sm_begin() const {
        return sectors_map.begin();
    }

    sm_const_iterator sm_end() const {
        return sectors_map.end();
    }

    int sm_width() const {
        return sectors_map_width;
    }

    int sm_height() const {
        return sectors_map_height;
    }

    unsigned long long last_kernel_time() const {
        return kernel_time;
    }

    void run(int steps) {
        pthread_mutex_lock(&steps_mutex);
        steps_shared = steps;
        pthread_cond_signal(&step_cv);
        pthread_mutex_unlock(&steps_mutex);
    }

    bool is_done() {
        pthread_mutex_lock(&steps_mutex);
        bool local_done = done;
        pthread_mutex_unlock(&steps_mutex);
        return local_done;
    }

    void set_end(int i, std::pair<float, float> p) {
        ends[i] = p;
        updated_ends = true;
    }
};
