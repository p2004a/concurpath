#include <thread>
#include <chrono>
#include <string>
#include <memory>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "display.h"
#include "map.h"

using namespace std;

int main(int argc, char *argv[]) {
    display::init();

    if (argc < 2) {
        fprintf(stderr, "USAGE\n\t%s map.png\n", argv[0]);
        return 1;
    }

    map m(argv[1]);

    display dis(800, 600);
    dis.run([&] (int width, int height) {
        m.set_screen_size(width, height);
        m.draw();
    });
    return 0;
}
