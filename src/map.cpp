#include <string>
#include <algorithm>
#include <memory>
#include <stdexcept>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "map.h"

using namespace std;

map::map(string filename) : bitmap(nullptr) {
    if (!(bitmap = al_load_bitmap(filename.c_str()))) {
        throw std::runtime_error("Cannot load map file");
    }
    width = al_get_bitmap_width(bitmap);
    height = al_get_bitmap_height(bitmap);
    data = make_unique<bool[]>(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char r, g, b;
            auto c = al_get_pixel(bitmap, x, y);
            al_unmap_rgb(c, &r, &g, &b);
            data[y * width + x] = r != 0;
        }
    }
}

map::~map() {
    if (bitmap) {
        al_destroy_bitmap(bitmap);
    }
}

void map::draw() {
    if (resized) {
        resized = false;

        auto old_target_bitmap = al_get_target_bitmap();
        if (render != nullptr) {
            al_destroy_bitmap(render);
        }
        render = al_create_bitmap(width * scale + 1, height * scale + 1);
        if (!render) {
            throw runtime_error("failed to create bitmap");
        }
        al_set_target_bitmap(render);

        auto black = al_map_rgb(0, 0, 0);
        auto lightgrey = al_map_rgb(200, 200, 200);
        auto grey = al_map_rgb(35, 35, 35);

        al_clear_to_color(black);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                al_draw_filled_rectangle(
                    x * scale, y * scale,
                    x * scale + scale, y * scale + scale,
                    (*this)[y][x] ? lightgrey : black
                );
            }
        }

        if (scale > 15) {
            for (int y = 0; y <= height; ++y) {
                al_draw_line(
                    0, y * scale,
                    width * scale, y * scale,
                    grey, 1.0
                );
            }
            for (int x = 0; x <= width; ++x) {
                al_draw_line(
                    x * scale, 0,
                    x * scale, height * scale,
                    grey, 1.0
                );
            }
        }

        al_set_target_bitmap(old_target_bitmap);
    }
    al_draw_bitmap(render, 0.0, 0.0, 0);
}

void map::set_screen_size(int _screen_width, int _screen_height) {
    if (screen_width != _screen_width || screen_height != _screen_height) {
        screen_width = _screen_width;
        screen_height = _screen_height;
        scale = min((double)screen_width / width,
                    (double)screen_height / height);
        resized = true;
    }
}
