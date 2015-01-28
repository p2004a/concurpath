#pragma once

/*
    Unfortunatelly I have to use pthreads not c++11 threads because
    nvcc doesn't support C++11 yet and I don't wan't to contrive
*/

#include <pthread.h>

#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class simulation {
    thrust::host_vector<thrust::pair<float, float> > units;
    thrust::host_vector<bool> map;
    int map_width, map_height;

    pthread_t thread;
    pthread_mutex_t steps_mutex;
    pthread_cond_t step_cv;
    volatile bool done;
    volatile int steps_shared;

    typedef thrust::host_vector<thrust::pair<float, float> >::const_iterator const_iterator;

    void thread_func();
    static void* thread_func_helper(void *context) {
        ((simulation *)context)->thread_func();
        return NULL;
    }

    simulation(simulation const &) {};
  public:

    template<typename It, typename Arr2D>
    simulation(It begin, It end, Arr2D const& arr, int w, int h)
      : units(begin, end), map(w * h), map_width(w), map_height(h), steps_shared(0) {
        #pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                map[y * w + x] = arr[y][x];
            }
        }

        pthread_mutex_init(&steps_mutex, NULL);
        pthread_cond_init(&step_cv, NULL);

        int res = pthread_create(&thread, NULL, &simulation::thread_func_helper, (void *)this);
        if (res) {
            throw std::runtime_error("failed to create thread");
        }
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

    const_iterator begin() const {
        return units.begin();
    }

    const_iterator end() const {
        return units.end();
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
};
