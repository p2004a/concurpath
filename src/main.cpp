#include <thread>
#include <chrono>
#include <string>
#include <memory>
#include <vector>

#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

#include "display.h"
#include "map.h"
#include "simulation.h"

using namespace std;

int main(int argc, char *argv[]) {
    display::init();

    if (argc < 2) {
        fprintf(stderr, "USAGE\n\t%s map.png\n", argv[0]);
        return 1;
    }

    map m(argv[1]);

    vector<pair<float, float>> units = {{0,0}, {3,4}, {6,7}};

    simulation s(units.begin(), units.end(), m, m.get_width(), m.get_height());

    display dis(800, 600);

    dis.run([&] (int width, int height) {
        m.set_screen_size(width, height);

        s.run(20);
        m.draw();
    });
    return 0;
}
