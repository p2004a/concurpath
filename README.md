concurpath
==========

Concurrent pathfinder

Build
-----

Dependencies:

- allegro5
- openmp supporting compiler
- cuda capable device and header files

CMake is a build system to build project, so just:

```sh
mkdir build; cd build
cmake ..
make
```

After that you should have `path` binary file in `build/` directory.

Running
-------

`path` binary takes three arguments

1. n - number of units to generate. Generated units are placed in random
    positions and are given random destinations.
2. spf - (simulations per frame) value is describing how many simulations
   to run per one rendering frame. Display is rendering in 60fps so setting
   spf to 10 will result in at most 600 simulations per second.
3. map.png - is a png file built from black and white pixels describing map.
   Black pixels are free space and white ones are walls. Sample maps are in
   `sample_maps` directory.

Examples:

    ./path 10 5 ../sample_maps/map05.png
    ./path 500 12 ../sample_maps/map02.png
