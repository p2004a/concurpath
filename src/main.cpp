#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>

#include "display.h"
#include "map.h"
#include "fpscounter.h"
#include "simulation.h"

using namespace std;

int main(int argc, char *argv[]) {
    display::init();

    auto font_average_mono_20 = al_load_ttf_font("fonts/AverageMono.ttf", -20, 0);
    if (font_average_mono_20 == NULL) {
        throw runtime_error("Cannot load font fonts/AverageMono.ttf size 20");
    }

    auto font_average_mono_12 = al_load_ttf_font("fonts/AverageMono.ttf", -14, 0);
    if (font_average_mono_12 == NULL) {
        throw runtime_error("Cannot load font fonts/AverageMono.ttf size 12");
    }

    if (argc < 4) {
        fprintf(stderr, "USAGE\n\t%s n spf map.png\n", argv[0]);
        return 1;
    }

    map m(argv[3]);

    display dis(800, 600);

    auto green = al_map_rgb(0, 200, 0);
    auto yellow = al_map_rgb(255, 255, 0);
    auto white = al_map_rgb(255, 255, 255);
    auto red = al_map_rgb(255, 0, 0);

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

    auto n = stoul(argv[1]);
    auto spf = stoul(argv[2]); // simulations pef frame
    vector<thrust::pair<float, float>> units(n);
    vector<thrust::pair<float, float>> ends(n);
    std::generate(units.begin(), units.end(), gen_rand_pos);

    simulation s(units.begin(), units.end(), m, m.get_width(), m.get_height());
    vector<thrust::pair<int, int>> sectors_map(s.sm_width() * s.sm_height());

    for (unsigned i = 0; i < n; ++i) {
        auto pos = gen_rand_pos();
        s.set_end(i, pos);
        ends[i] = pos;
    }

    unique_ptr<ALLEGRO_VERTEX[]> unit_pixels(nullptr);
    if (n >= 4000) {
        unit_pixels = unique_ptr<ALLEGRO_VERTEX[]>(new ALLEGRO_VERTEX[n]);
    }

    fps_counter fps_display;
    fps_counter fps_simulation(300);
    unsigned long long last_kernel_time = 0;
    dis.run([&] (int width, int height) {
        fps_display.tick();
        if (s.is_done()) {
            fps_simulation.tick();
            last_kernel_time = s.last_kernel_time();
            units.clear();
            sectors_map.clear();
            std::copy(s.u_begin(), s.u_end(), back_inserter(units));
            std::copy(s.sm_begin(), s.sm_end(), back_inserter(sectors_map));

            for (unsigned i = 0; i < n; ++i) {
                if ((int)units[i].first == (int)ends[i].first
                  && (int)units[i].second == (int)ends[i].second) {
                    {
                        auto pos = gen_rand_pos();
                        s.set_end(i, pos);
                        ends[i] = pos;
                    }
                }
            }

            for (unsigned i = 0; i < n; ++i) {
                if (!isnormal(units[i].first) || !isnormal(units[i].second)) {
                    printf("%f %f\n", units[i].first, units[i].second);
                    throw logic_error("one of units doesnt have correct coordinares ");
                }
            }

            s.run(spf);
        }

        m.set_screen_size(width, height);
        float scale = m.get_scale();

        m.draw();
        if (n < 4000) {
            for (unsigned i = 0; i < n; ++i) {
                auto p = units[i];
                auto e = ends[i];
                if (0.2 * scale < 1.0) {
                    al_draw_pixel(p.first * scale, p.second * scale, green);
                } else {
                    al_draw_filled_circle(p.first * scale, p.second * scale, 0.2 * scale, green);
                    if (n <= 64) {
                        al_draw_line(p.first * scale, p.second * scale, e.first * scale, e.second * scale, yellow, 1.0);
                    }
                }
            }
        } else {
            for (unsigned i = 0; i < n; ++i) {
                auto const& p = units[i];
                auto & pixel = unit_pixels[i];
                pixel.z = 0;
                pixel.x = p.first * scale;
                pixel.y = p.second * scale;
                pixel.color = green;
            }
            al_draw_prim(unit_pixels.get(), NULL, NULL, 0, n, ALLEGRO_PRIM_POINT_LIST);
        }

        if (s.sm_height() < 25 && s.sm_width() < 25) {
            for (int y = 0; y < s.sm_height(); ++y) {
                for (int x = 0; x < s.sm_width(); ++x) {
                    al_draw_textf(font_average_mono_12, red, x * scale * MAP_SECTOR_SIZE, y * scale * MAP_SECTOR_SIZE, ALLEGRO_ALIGN_LEFT, "%d", sectors_map[y * s.sm_width() + x].second);
                }
            }
        }

        al_draw_textf(font_average_mono_20, white, width - 10, 10, ALLEGRO_ALIGN_RIGHT, "%.2f fps", fps_display.get());
        al_draw_textf(font_average_mono_20, white, width - 10, 40, ALLEGRO_ALIGN_RIGHT, "%.2f sps", fps_simulation.get() * spf);
        al_draw_textf(font_average_mono_20, white, width - 10, 70, ALLEGRO_ALIGN_RIGHT, "%.2fus", (double)last_kernel_time / 1000);
    });
    return 0;
}
