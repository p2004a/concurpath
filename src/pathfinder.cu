#include <cstdio>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "pathfinder.h"

void pathfinder::thread_func() {
    const int THREADS_PER_BLOCK = 128;

    thrust::device_vector<bool> d_map(map.begin(), map.end());

    pthread_mutex_lock(&que_mutex);
    pthread_cond_signal(&que_cv);
    pthread_mutex_unlock(&que_mutex);

    while (true) {
        pathfinder_future *future;
        thrust::pair<int, int> end;

        pthread_mutex_lock(&que_mutex);
        while (que.empty()) {
            pthread_cond_wait(&que_cv, &que_mutex);
        }
        end = que.front();
        que.pop_front();
        future = futures_que.front();
        futures_que.pop_front();
        pthread_mutex_unlock(&que_mutex);

        if (end == thrust::pair<int, int>(-1, -1)) {
            break;
        }

        thrust::device_vector<thrust::pair<int, int> > result;
        
    }
    pthread_exit(NULL);
}
