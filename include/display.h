#pragma once

#include <stdexcept>
#include <functional>
#include <allegro5/allegro.h>

class display {
    ALLEGRO_DISPLAY *dis;
    ALLEGRO_EVENT_QUEUE *event_queue;

  public:
    display(int width_, int height_);
    display (display const&) = delete;
    display & operator=(display const&) = delete;
    ~display();

    template<class Func>
    void run(Func f, int fps = 60);
};

template<class Func>
void display::run(Func f, int fps) {
    auto timer = al_create_timer(1.0 / fps);
    if (!timer) {
        throw std::runtime_error("failed to create timer");
    }
    al_register_event_source(event_queue, al_get_timer_event_source(timer));

    int width = al_get_display_width(dis);
    int height = al_get_display_height(dis);
    ALLEGRO_BITMAP *backbuffer = al_get_backbuffer(dis);

    bool end = false;
    bool redraw = true;

    al_start_timer(timer);
    while (!end) {
        ALLEGRO_EVENT ev;
        al_wait_for_event(event_queue, &ev);

        switch (ev.type) {
            case ALLEGRO_EVENT_KEY_DOWN:
                if (ev.keyboard.keycode == ALLEGRO_KEY_ESCAPE) {
                    end = true;
                }
                break;

            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                end = true;
                break;

            case ALLEGRO_EVENT_DISPLAY_RESIZE:
                al_acknowledge_resize(dis);
                width = ev.display.width;
                height = ev.display.height;
                break;

            case ALLEGRO_EVENT_TIMER:
                redraw = true;
                break;
        }

        if (redraw && al_is_event_queue_empty(event_queue)) {
            redraw = false;
            al_set_target_bitmap(backbuffer);
            al_clear_to_color(al_map_rgb(0,0,0));
            f(width, height);
            al_flip_display();
        }
    }

    al_destroy_timer(timer);
}
