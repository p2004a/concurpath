#include <thread>
#include <chrono>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "display.h"
#include "map.h"
#include "simulation.h"

using namespace std;

int main(int argc, char *argv[]) {
    display::init();

    if (argc < 2) {
        fprintf(stderr, "USAGE\n\t%s map.png\n", argv[0]);
        return 1;
    }

    map m(argv[1]);

    display dis(800, 600);

    auto green = al_map_rgb(0, 255, 0);

    random_device rd;
    default_random_engine re(rd());
    uniform_int_distribution<int> x_rand(0, m.get_width() - 1);
    uniform_int_distribution<int> y_rand(0, m.get_height() - 1);
    std::uniform_real_distribution<> diff(0.10, 0.90);

    auto gen_rand_pos = [&]() {
        while (true) {
            int x = x_rand(re);
            int y = y_rand(re);
            if (!m[y][x]) {
                return make_pair((float)x + diff(re), (float)y + diff(re));
            }
        }
    };

    int n = 20;
    vector<thrust::pair<float, float>> units(n);
    vector<thrust::pair<float, float>> ends(n);
    std::generate(units.begin(), units.end(), gen_rand_pos);

    simulation s(units.begin(), units.end(), m, m.get_width(), m.get_height());

    for (int i = 0; i < n; ++i) {
        auto pos = gen_rand_pos();
        s.set_end(i, pos);
        ends[i] = pos;
    }

    dis.run([&] (int width, int height) {
        if (s.is_done()) {
            units.clear();
            std::copy(s.begin(), s.end(), back_inserter(units));

            for (int i = 0; i < n; ++i) {
                if ((int)units[i].first == (int)ends[i].first
                  && (int)units[i].second == (int)ends[i].second) {
                    {
                        auto pos = gen_rand_pos();

                        s.set_end(i, pos);
                        ends[i] = pos;
                    }
                }
            }

            s.run(20);
        } else {
            printf("dropped frame\n");
        }

        m.set_screen_size(width, height);
        float scale = m.get_scale();

        m.draw();
        for (auto p: units) {
            if (0.2 * scale < 1.0) {
                al_draw_pixel(p.first * scale, p.second * scale, green);
            } else {
                al_draw_filled_circle(p.first * scale, p.second * scale, 0.2 * scale, green);
            }
        }
    });
    return 0;
}
