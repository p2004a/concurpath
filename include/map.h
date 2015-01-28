#pragma once

#include <string>
#include <memory>

#include <allegro5/allegro.h>

class map {
    ALLEGRO_BITMAP *bitmap, *render = nullptr;
    int width, height;
    std::unique_ptr<bool[]> data;
    int screen_width, screen_height;
    float scale;
    bool resized = true;

  public:

    map(map const&) = delete;
    map(std::string filename);
    ~map();
    void draw();
    void set_screen_size(int _screen_width, int _screen_height);

    int get_width() const {
        return width;
    }

    int get_height() const {
        return height;
    }

    float get_scale() const {
        return scale;
    }

    bool const* operator[] (int idx) {
        return data.get() + idx * width;
    }
};
