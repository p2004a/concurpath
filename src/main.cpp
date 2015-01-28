#include <thread>
#include <chrono>
#include <string>
#include <memory>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "display.h"

using namespace std;

class map {
    ALLEGRO_BITMAP *bitmap;
    int width, height;
    unique_ptr<bool[]> data;

  public:

    map(map const&) = delete;

    map(string filename) : bitmap(nullptr) {
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

    ~map() {
        if (bitmap) {
            al_destroy_bitmap(bitmap);
        }
    }

    int get_width() const {
        return width;
    }

    int get_height() const {
        return height;
    }

    bool const* operator[] (int idx) {
        return data.get() + idx * width;
    }
};

int main(int argc, char *argv[]) {
    display::init();

    if (argc < 2) {
        fprintf(stderr, "USAGE\n\t%s map.png\n", argv[0]);
        return 1;
    }

    map m(argv[1]);

    display dis(800, 600);
    dis.run([&] (int width, int height) {
        auto blue = al_map_rgb(0, 255, 0);
        al_draw_circle(width / 2.0, height / 2.0, 20.0, blue, 5.0);

        auto grey = al_map_rgb(22,22,22);
        auto white = al_map_rgb(255,255,255);
        for (int y = 0; y < m.get_height(); ++y) {
            for (int x = 0; x < m.get_width(); ++x) {
                al_draw_filled_rectangle(x * 5, y * 5, x * 5 + 5, y * 5 + 5, m[y][x] ? white : grey);
            }
        }
    });
    return 0;
}
