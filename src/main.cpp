#include <thread>
#include <chrono>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "display.h"

using namespace std;

int main() {
    display dis(800, 600);
    dis.run([] (int width, int height) {
        auto blue = al_map_rgb(0, 255, 0);
        al_draw_circle(width / 2.0, height / 2.0, 20.0, blue, 5.0);
    });
    return 0;
}
