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
#include "pathfinder.h"

#define UNUSED __attribute__((__unused__))

int main(int argc, char *argv[]) {
    display::init();

    auto font_average_mono_20 = al_load_ttf_font("fonts/AverageMono.ttf", -20, 0);
    if (font_average_mono_20 == NULL) {
        throw std::runtime_error("Cannot load font fonts/AverageMono.ttf size 20");
    }

    if (argc < 2) {
        fprintf(stderr, "USAGE\n\t%s map.png\n", argv[0]);
        return 1;
    }

    map m(argv[1]);

    display dis(800, 600);

    auto green = al_map_rgb(0, 200, 0);
    auto white = al_map_rgb(255, 255, 255);
    auto red = al_map_rgb(255, 0, 0);
    auto o_blue = al_map_rgb(0, 0, 255);
    auto o_yellow = al_map_rgb(255, 255, 0);

    auto mouse_pos = std::make_pair(-1, -1);
    auto mouse_begin = std::make_pair(-1, -1);
    auto map_mouse_begin = std::make_pair(0, 0);
    auto map_mouse_pos = std::make_pair(0, 0);

    dis.mouse_up([&] (int x, int y, unsigned button UNUSED) {
        mouse_begin = std::make_pair(x, y);
    });

    dis.mouse_move([&] (int x, int y) {
        mouse_pos = std::make_pair(x, y);
    });

    thrust::host_vector<bool> map_vector(m.get_height() * m.get_width());
    thrust::host_vector<int> out_vector(m.get_height() * m.get_width());
    for (int y = 0; y < m.get_height(); ++y) {
        for (int x = 0; x < m.get_width(); ++x) {
            map_vector[y * m.get_width() + x] = m[y][x];
        }
    }
    thrust::device_vector<bool> d_map_vector(map_vector.begin(), map_vector.end());

    fps_counter fps_display;
    dis.run([&] (int width, int height) {
        fps_display.tick();

        m.set_screen_size(width, height);
        float scale = m.get_scale();

        if (mouse_begin != std::make_pair(-1, -1)) {
            auto new_map_mouse_begin = std::make_pair(mouse_begin.first / scale, mouse_begin.second / scale);
            if (new_map_mouse_begin.first < m.get_width() && new_map_mouse_begin.second < m.get_height()) {
                map_mouse_begin = new_map_mouse_begin;
            }
            mouse_begin = std::make_pair(-1, -1);
        }

        if (mouse_pos != std::make_pair(-1, -1)) {
            auto new_map_mouse_pos = std::make_pair(mouse_pos.first / scale, mouse_pos.second / scale);
            if (new_map_mouse_pos.first < m.get_width() && new_map_mouse_pos.second < m.get_height()) {
                map_mouse_pos = new_map_mouse_pos;
            }
            mouse_begin = std::make_pair(-1, -1);
        }

        m.draw();

        if (map_mouse_pos != map_mouse_begin) {
            auto c = green;
            if (!line_of_sight_cpu(map_mouse_begin, map_mouse_pos, map_vector.begin(), m.get_width(), m.get_height(), out_vector)) {
                c = red;
            }
            for (int y = 0; y < m.get_height(); ++y) {
                for (int x = 0; x < m.get_width(); ++x) {
                    if (out_vector[y * m.get_width() + x]) {
                        al_draw_rectangle(
                            (x + 0.1) * scale, (y + 0.1) * scale,
                            (x - 0.1) * scale + scale, (y - 0.1) * scale + scale,
                            out_vector[y * m.get_width() + x] == 1 ? o_blue : o_yellow,
                            1.0
                        );
                    }
                }
            }
            al_draw_line((map_mouse_begin.first + 0.5) * scale,
                         (map_mouse_begin.second + 0.5) * scale,
                         (map_mouse_pos.first + 0.5) * scale,
                         (map_mouse_pos.second + 0.5) * scale,
                         c, 1.0);
        }

        al_draw_textf(font_average_mono_20, white, width - 10, 10, ALLEGRO_ALIGN_RIGHT, "%.2f fps", fps_display.get());
    });
    return 0;
}
