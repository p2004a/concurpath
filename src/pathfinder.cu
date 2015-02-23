#include <cstdio>
#include <cstdlib>
#include <time.h>

#include <algorithm>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/logical.h>
#include <thrust/swap.h>
#include <thrust/fill.h>

#include "pathfinder.h"

#define _IDX(x, y) ((int)(y) * map_width + (int)(x))
#define IDX(x, y, ...) ((__VA_ARGS__ + 0) ? _IDX(y, x) : _IDX(x, y))

class line_of_sight_functor {
  public:
    __host__ __device__
    bool operator()(
        thrust::tuple<
            int,
            thrust::pair<int, int>,
            thrust::pair<int, int>,
            thrust::device_vector<bool>::iterator,
            int
#ifdef LINE_OF_SIGHT_DEBUG
            ,thrust::device_vector<int>::iterator
#endif
        > data
    ) const {
        int i = thrust::get<0>(data);
        thrust::pair<int, int> begin = thrust::get<1>(data);
        thrust::pair<int, int> end = thrust::get<2>(data);
        thrust::device_vector<bool>::iterator map = thrust::get<3>(data);
        int map_width = thrust::get<4>(data);

#ifdef LINE_OF_SIGHT_DEBUG
        thrust::device_vector<int>::iterator out = thrust::get<5>(data);
#endif

        int w = end.first - begin.first;
        int h = end.second - begin.second;

        bool s = abs(w) > abs(h);
        if (s) {
            thrust::swap(w, h);
            thrust::swap(begin.first, begin.second);
            thrust::swap(end.first, end.second);
        }

        int x_sign = 1;
        int y_sign = 1;
        if (w != 0) {
            x_sign = w / abs(w);
        }
        if (h != 0) {
            y_sign = h / abs(h);
        }
        w = w * x_sign;
        h = h * y_sign;

        int yd = i * y_sign;
        int xd = (w * i) / h * x_sign;

        double line_top_x = (((i - 0.5) * w) / h + 0.5) * x_sign;
        double line_bottom_x = (((i + 0.5) * w) / h + 0.5) * x_sign;
        double corner_x = xd + x_sign;

        bool result;

#ifdef LINE_OF_SIGHT_DEBUG
        out[IDX(begin.first + xd, begin.second + yd, s)] = fabs(line_top_x) <= fabs(corner_x) ? 1 : 2;
        out[IDX(begin.first + xd + x_sign, begin.second + yd, s)] = fabs(line_bottom_x) >= fabs(corner_x) ? 1 : 2;
#endif
        result = (fabs(line_top_x) <= fabs(corner_x)
                  && map[IDX(begin.first + xd, begin.second + yd, s)])
              || (fabs(line_bottom_x) >= fabs(corner_x)
                  && map[IDX(begin.first + xd + x_sign, begin.second + yd, s)]);
        return result;
    }
};

bool line_of_sight_gpu(
    thrust::pair<int, int> begin,
    thrust::pair<int, int> end,
    thrust::device_vector<bool>::iterator map,
    int map_width,
    int map_height
#ifdef LINE_OF_SIGHT_DEBUG
    ,thrust::host_vector<int> &out
#endif
) {
    assert(begin.first >= 0 && begin.first < map_width && begin.second >= 0 && begin.second < map_height);
    assert(end.first >= 0 && end.first < map_width && end.second >= 0 && end.second < map_height);

    thrust::constant_iterator<thrust::pair<int, int> > begin_iter(begin);
    thrust::constant_iterator<thrust::pair<int, int> > end_iter(end);
    thrust::constant_iterator<thrust::device_vector<bool>::iterator> map_iter(map);
    thrust::constant_iterator<int> map_width_iter(map_width);

#ifdef LINE_OF_SIGHT_DEBUG
    assert(out.size() == map_width * map_height);
    thrust::device_vector<int> d_out(out.begin(), out.end());
    thrust::fill(d_out.begin(), d_out.end(), 0);
    thrust::constant_iterator<thrust::device_vector<int>::iterator> out_iter(d_out.begin());
#endif

    int n = max(abs(begin.first - end.first), abs(begin.second - end.second)) + 1;

    bool result = thrust::none_of(
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0),
            begin_iter,
            end_iter,
            map_iter,
            map_width_iter
#ifdef LINE_OF_SIGHT_DEBUG
            ,out_iter
#endif
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(n),
            begin_iter,
            end_iter,
            map_iter,
            map_width_iter
#ifdef LINE_OF_SIGHT_DEBUG
            ,out_iter
#endif
        )),
        line_of_sight_functor()
    );

#ifdef LINE_OF_SIGHT_DEBUG
    thrust::copy(d_out.begin(), d_out.end(), out.begin());
#endif

    return result;
}

bool line_of_sight_cpu(
    thrust::pair<int, int> begin,
    thrust::pair<int, int> end,
    thrust::host_vector<bool>::iterator map,
    int map_width,
    int map_height
#ifdef LINE_OF_SIGHT_DEBUG
    ,thrust::host_vector<int> &out
#endif
) {
#ifdef LINE_OF_SIGHT_DEBUG
    thrust::fill(out.begin(), out.end(), 0);
#endif

    const bool s = (fabs(end.second - begin.second) > fabs(end.first - begin.first));
    if (s) {
        thrust::swap(begin.first, begin.second);
        thrust::swap(end.first, end.second);
    }

    if (begin.first > end.first) {
        thrust::swap(begin, end);
    }

    const float dx = end.first - begin.first;
    const float dy = fabs(end.second - begin.second);

    float error = dx / 2.0f;
    const int ystep = (begin.second < end.second) ? 1 : -1;
    int y = (int)begin.second;

    for (int x = (int)begin.first; x <= (int)end.first; x++) {
#ifdef LINE_OF_SIGHT_DEBUG
        out[IDX(x, y, s)] = 1;
        if (map[IDX(x, y, s)]) {
            return false;
        }
#endif

        error -= dy;
        if (error < -(dy / 2)) {
            y += ystep;
#ifdef LINE_OF_SIGHT_DEBUG
            out[IDX(x, y, s)] = 1;
            if (map[IDX(x, y, s)]) {
                return false;
            }
#endif
            error += dx;
        }
    }

    return true;
}

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
