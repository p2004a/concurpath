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
