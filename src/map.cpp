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

        unique_ptr<ALLEGRO_VERTEX[]> map_pixels = unique_ptr<ALLEGRO_VERTEX[]>(new ALLEGRO_VERTEX[height * width * 6]);

        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = (y * width * 6) + (x * 6);
                ALLEGRO_COLOR c = (*this)[y][x] ? lightgrey : black;
                map_pixels[i+0].x = x * scale;
                map_pixels[i+0].y = y * scale;
                map_pixels[i+0].z = 0.0;
                map_pixels[i+0].color = c;
                map_pixels[i+1].x = x * scale + scale;
                map_pixels[i+1].y = y * scale;
                map_pixels[i+1].z = 0.0;
                map_pixels[i+1].color = c;
                map_pixels[i+2].x = x * scale;
                map_pixels[i+2].y = y * scale + scale;
                map_pixels[i+2].z = 0.0;
                map_pixels[i+2].color = c;
                map_pixels[i+3].x = x * scale + scale;
                map_pixels[i+3].y = y * scale + scale;
                map_pixels[i+3].z = 0.0;
                map_pixels[i+3].color = c;
                map_pixels[i+4] = map_pixels[i+1];
                map_pixels[i+5] = map_pixels[i+2];
            }
        }
        al_draw_prim(map_pixels.get(), NULL, NULL, 0, height * width * 6, ALLEGRO_PRIM_TRIANGLE_LIST);

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
