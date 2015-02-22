#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <time.h>

#include <algorithm>
#include <set>

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

#define INF 2000000000

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
        int map_wdth = thrust::get<4>(data);

#ifdef LINE_OF_SIGHT_DEBUG
        thrust::device_vector<int>::iterator out = thrust::get<5>(data);
#endif

        int w = end.first - begin.first;
        int h = end.second - begin.second;
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

        if (w > h) {
            // more horizontal
            int xd = i * x_sign;
            int yd = (h * i) / w * y_sign;

            double line_left_y = (((i - 0.5) * h) / w + 0.5) * y_sign;
            double line_right_y = (((i + 0.5) * h) / w + 0.5) * y_sign;
            double corner_y = yd + y_sign;

#ifdef LINE_OF_SIGHT_DEBUG
            out[(begin.second + yd) * map_wdth + (begin.first + xd)] = fabs(line_left_y) <= fabs(corner_y) ? 1 : 2;
            out[(begin.second + (yd + y_sign)) * map_wdth + (begin.first + xd)] = fabs(line_right_y) >= fabs(corner_y) ? 1 : 2;
#endif

            return (fabs(line_left_y) <= fabs(corner_y)
                    && map[(begin.second + yd) * map_wdth + (begin.first + xd)])
                || (fabs(line_right_y) >= fabs(corner_y)
                    && map[(begin.second + (yd + y_sign)) * map_wdth + (begin.first + xd)]);
        } else {
            // more vertical
            int yd = i * y_sign;
            int xd = (w * i) / h * x_sign;

            double line_top_x = (((i - 0.5) * w) / h + 0.5) * x_sign;
            double line_bottom_x = (((i + 0.5) * w) / h + 0.5) * x_sign;
            double corner_x = xd + x_sign;

#ifdef LINE_OF_SIGHT_DEBUG
            out[(begin.second + yd) * map_wdth + (begin.first + xd)] = fabs(line_top_x) <= fabs(corner_x) ? 1 : 2;
            out[(begin.second + yd) * map_wdth + (begin.first + (xd + x_sign))] = fabs(line_bottom_x) >= fabs(corner_x) ? 1 : 2;
#endif

            return (fabs(line_top_x) <= fabs(corner_x)
                    && map[(begin.second + yd) * map_wdth + (begin.first + xd)])
                || (fabs(line_bottom_x) >= fabs(corner_x)
                    && map[(begin.second + yd) * map_wdth + (begin.first + (xd + x_sign))]);
        }
    }
};

bool line_of_sight(
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

class theta_comp {
    thrust::host_vector<double> &dist;

  public:
    theta_comp(thrust::host_vector<double> &_dist) : dist(_dist) {}

    bool operator() (int a, int b) const {
        if (dist[a] < dist[b]) {
            return true;
        } else if (dist[a] > dist[b]) {
            return false;
        }
        return a < b;
    }
};

void pathfinder::thread_func() {
    thrust::device_vector<bool> d_map(map.begin(), map.end());
    thrust::host_vector<double> dist(map.size());
    thrust::host_vector<double> h_dist(map.size());
    thrust::host_vector<int> parent(map.size());

    const int nlen = 8;
    double sqrt2 = sqrt(2.0);
    int ndx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int ndy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    double ndist[8] = {sqrt2, 1, sqrt2, 1, 1, sqrt2, 1, sqrt2};

    while (true) {
        pathfinder_future *future;
        thrust::pair<int, int> begin, end;

        pthread_mutex_lock(&que_mutex);
        while (end_que.empty()) {
            pthread_cond_wait(&que_cv, &que_mutex);
        }
        end = end_que.front();
        begin = begin_que.front();
        end_que.pop_front();
        begin_que.pop_front();
        future = futures_que.front();
        futures_que.pop_front();
        pthread_mutex_unlock(&que_mutex);

        if (end == thrust::pair<int, int>(-1, -1)) {
            break;
        }

        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        int begin_idx = begin.second * map_width + begin.first;
        int end_idx = end.second * map_width + end.first;

        thrust::fill(dist.begin(), dist.end(), INF);
        thrust::fill(h_dist.begin(), h_dist.end(), INF);
        theta_comp comp(h_dist);
        std::set<int, theta_comp> q(comp);

        dist[begin_idx] = 0;
        h_dist[begin_idx] = 0;
        parent[begin_idx] = begin_idx;
        q.insert(begin_idx);

        int num = 0, lsc = 0;
        while (!q.empty()) {
            ++num;
            int idx = *(q.begin());
            thrust::pair<int, int> pos = idx_to_pair(idx);
            q.erase(q.begin());
            int pidx = parent[idx];
            if (idx == end_idx) {
                break;
            }
            for (int i = 0; i < nlen; ++i) {
                thrust::pair<int, int> cpos = thrust::make_pair(pos.first + ndx[i], pos.second + ndy[i]);
                int cidx = cpos.second * map_width + cpos.first;
                if (cpos.second >= 0 && cpos.second < map_height && cpos.first >= 0 && cpos.first < map_width && !map[cidx]) {
                    if (ndx[i] && ndy[i] && (map[cidx - ndx[i]] || map[cidx - map_width * ndy[i]])) {
                        continue;
                    }
                    double d;
                    int p;
                    ++lsc;
                    if (line_of_sight(idx_to_pair(cidx), idx_to_pair(pidx), d_map.begin(), map_width, map_height)) {
                        d = dist[pidx] + idx_distance(pidx, cidx);
                        p = pidx;
                    } else {
                        d = dist[idx] + ndist[i];
                        p = idx;
                    }
                    if (d < dist[cidx]) {
                        q.erase(cidx);
                        parent[cidx] = p;
                        dist[cidx] = d;
                        h_dist[cidx] = d + idx_distance(end_idx, cidx);
                        q.insert(cidx);
                    }
                }
            }
        }

        thrust::host_vector<thrust::pair<int, int> > result;
        for (int idx = end_idx; idx != parent[idx]; idx = parent[idx]) {
            result.push_back(idx_to_pair(idx));
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);
        unsigned long long diff = (long long)(t2.tv_sec - t1.tv_sec) * 1000LL + (t2.tv_nsec - t1.tv_nsec) / 1000000;

        printf("vis: %d lsc: %d seg: %d time: %dms\n", num, lsc, result.size(), diff);

        future->set_result(result);
    }
    pthread_exit(NULL);
}
