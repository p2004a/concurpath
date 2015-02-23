#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <time.h>

#include <algorithm>
#include <set>

#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <thrust/fill.h>

#include "pathfinder.h"

#define INF 2000000000

#define _IDX(x, y) ((int)(y) * map_width + (int)(x))
#define IDX(x, y, ...) ((__VA_ARGS__ + 0) ? _IDX(y, x) : _IDX(x, y))

bool line_of_sight(
    thrust::pair<int, int> begin,
    thrust::pair<int, int> end,
    thrust::host_vector<bool>::iterator map,
    int map_width,
    int map_height
#ifdef LINE_OF_SIGHT_DEBUG
    ,thrust::host_vector<int> &out
#endif
) {
    assert(begin.first >= 0 && begin.first < map_width && begin.second >= 0 && begin.second < map_height);
    assert(end.first >= 0 && end.first < map_width && end.second >= 0 && end.second < map_height);

#ifdef LINE_OF_SIGHT_DEBUG
    thrust::fill(out.begin(), out.end(), 0);
#endif

    const bool s = (abs(end.second - begin.second) > abs(end.first - begin.first));
    if (s) {
        thrust::swap(begin.first, begin.second);
        thrust::swap(end.first, end.second);
    }

    if (begin.first > end.first) {
        thrust::swap(begin, end);
    }

    const float dx = end.first - begin.first;
    const float dy = abs(end.second - begin.second);

    float error = dx / 2.0f;
    const int ystep = (begin.second < end.second) ? 1 : -1;
    int y = (int)begin.second;

    for (int x = (int)begin.first; x <= (int)end.first; x++) {
#ifdef LINE_OF_SIGHT_DEBUG
        out[IDX(x, y, s)] = 1;
#endif
        if (map[IDX(x, y, s)]) {
            return false;
        }

        error -= dy;
        if (error < -(dy / 2)) {
            y += ystep;
#ifdef LINE_OF_SIGHT_DEBUG
            out[IDX(x, y, s)] = 1;
#endif
            if (map[IDX(x, y, s)]) {
                return false;
            }
            error += dx;
        }
    }

    return true;
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
                    if (line_of_sight(idx_to_pair(cidx), idx_to_pair(pidx), map.begin(), map_width, map_height)) {
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

        future->set_result(result);
    }
    pthread_exit(NULL);
}
