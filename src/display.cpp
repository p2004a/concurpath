#include <stdexcept>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_native_dialog.h>
#include <allegro5/allegro_image.h>

#include "display.h"

display::display(int width_, int height_) : dis(nullptr), event_queue(nullptr) {
    if(!al_init()) {
        throw std::runtime_error("failed to initialize allegro");
    }

#if ALLEGRO_SUB_VERSION >= 1 || (ALLEGRO_SUB_VERSION >= 0 && ALLEGRO_WIP_VERSION >= 9)
    if (!al_init_native_dialog_addon()) {
        throw std::runtime_error("failed to initialize native dialog addon");
    }
#endif

    if (!al_init_image_addon()) {
        throw std::runtime_error("failed to initialize allegro image addon");
    }

    if (!al_init_primitives_addon()) {
        throw std::runtime_error("failed to initialize allegro primitives");
    }

    if (!al_install_mouse()) {
        throw std::runtime_error("failed to initialize the mouse");
    }

    if (!al_install_keyboard()) {
        throw std::runtime_error("failed to initialize the keyboard");
    }

    al_set_new_display_option(ALLEGRO_VSYNC, 1, ALLEGRO_SUGGEST);
    al_set_new_display_option(ALLEGRO_SAMPLE_BUFFERS, 1, ALLEGRO_SUGGEST);
    al_set_new_display_option(ALLEGRO_SAMPLES, 4, ALLEGRO_SUGGEST);
    al_set_new_display_flags(ALLEGRO_WINDOWED | ALLEGRO_RESIZABLE);

    dis = al_create_display(width_, height_);
    if(!dis) {
        throw std::runtime_error("failed to create display");
    }

    event_queue = al_create_event_queue();
    if (!event_queue) {
        al_destroy_display(dis);
        throw std::runtime_error("failed to create event_queue");
    }

    al_register_event_source(event_queue, al_get_display_event_source(dis));
    al_register_event_source(event_queue, al_get_mouse_event_source());
    al_register_event_source(event_queue, al_get_keyboard_event_source());

    al_set_target_backbuffer(dis);
    al_clear_to_color(al_map_rgb(0,0,0));
    al_flip_display();
}

display::~display() {
    al_destroy_display(dis);
    al_destroy_event_queue(event_queue);
}
