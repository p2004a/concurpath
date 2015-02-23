#pragma once

//#define NDEBUG

#include <cassert>
#include <cmath>
#include <pthread.h>

#include <stdexcept>
#include <deque>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAP_SECTOR_SIZE 4

bool line_of_sight(
    thrust::pair<int, int> begin,
    thrust::pair<int, int> end,
    thrust::host_vector<bool>::iterator map,
    int map_width,
    int map_height
#ifdef LINE_OF_SIGHT_DEBUG
    ,thrust::host_vector<int> &out
#endif
);

class pathfinder_future {
    pthread_mutex_t mu;
    thrust::host_vector<thrust::pair<int, int> > result;
    volatile bool done;

  public:
    pathfinder_future() : done(false) {
        pthread_mutex_init(&mu, NULL);
    }

    ~pathfinder_future() {
        pthread_mutex_destroy(&mu);
    }

    void set_result(thrust::host_vector<thrust::pair<int, int> > _result) {
        result = _result;
        pthread_mutex_lock(&mu);
        done = true;
        pthread_mutex_unlock(&mu);
    }

    bool is_done() {
        pthread_mutex_lock(&mu);
        bool local_done = done;
        pthread_mutex_unlock(&mu);
        return local_done;
    }

    thrust::host_vector<thrust::pair<int, int> > get() const {
        return result;
    }
};

class pathfinder {
    thrust::host_vector<bool> map;
    int map_width, map_height;

    std::deque<thrust::pair<int, int> > begin_que, end_que;
    std::deque<pathfinder_future*> futures_que;

    pthread_t thread;
    pthread_mutex_t que_mutex;
    pthread_cond_t que_cv;
    unsigned long long kernel_time;

    void thread_func();
    static void* thread_func_helper(void *context) {
        ((pathfinder *)context)->thread_func();
        return NULL;
    }

    pathfinder(pathfinder const &) {};

    thrust::pair<int, int> idx_to_pair(int idx) {
        return thrust::make_pair(idx % map_width, idx / map_width);
    }

    double idx_distance(int a, int b) {
        return hypot((double)(a % map_width - b % map_width), (double)(a / map_width - b / map_width));
    };
  public:

    template<typename Arr2D>
    pathfinder(Arr2D const& arr, int w, int h)
      : map(w * h), map_width(w), map_height(h) {

        #pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                map[y * w + x] = arr[y][x];
            }
        }

        pthread_mutex_init(&que_mutex, NULL);
        pthread_cond_init(&que_cv, NULL);

        pthread_mutex_lock(&que_mutex);

        int res = pthread_create(&thread, NULL, &pathfinder::thread_func_helper, (void *)this);
        if (res) {
            pthread_mutex_unlock(&que_mutex);
            throw std::runtime_error("failed to create thread");
        }

        pthread_mutex_unlock(&que_mutex);
    }

    ~pathfinder() {
        find_path(thrust::make_pair(-1, -1), thrust::make_pair(-1, -1));

        int res = pthread_join(thread, NULL);
        if (res) {
            throw std::runtime_error("failed to join thread");
        }

        pthread_mutex_destroy(&que_mutex);
        pthread_cond_destroy(&que_cv);
    }

    pathfinder_future *find_path(thrust::pair<int, int> a, thrust::pair<int, int> b) {
        pathfinder_future *future = new pathfinder_future();
        pthread_mutex_lock(&que_mutex);
        begin_que.push_back(a);
        end_que.push_back(b);
        futures_que.push_back(future);
        pthread_cond_signal(&que_cv);
        pthread_mutex_unlock(&que_mutex);
        return future;
    }
};
