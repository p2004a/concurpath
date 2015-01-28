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
    std::uniform_real_distribution<> diff(0.05, 0.95);

    vector<thrust::pair<float, float>> units(20);
    std::generate(units.begin(), units.end(), [&]() {
        while (true) {
            int x = x_rand(re);
            int y = y_rand(re);
            if (!m[y][x]) {
                return make_pair(x + diff(re), y + diff(re));
            }
        }
    });

    simulation s(units.begin(), units.end(), m, m.get_width(), m.get_height());

    dis.run([&] (int width, int height) {
        m.set_screen_size(width, height);
        float scale = m.get_scale();

        if (s.is_done()) {
            units.clear();
            std::copy(s.begin(), s.end(), back_inserter(units));
            s.run(20);
        }

        m.draw();
        for (auto p: units) {
            al_draw_filled_circle(p.first * scale, p.second * scale, 0.2 * scale, green);
        }
    });
    return 0;
}
