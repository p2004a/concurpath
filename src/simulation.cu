#include <cstdio>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "simulation.h"

void simulation::thread_func() {
    while (true) {
        int steps;
        pthread_mutex_lock(&steps_mutex);
        while (steps_shared == 0) {
            pthread_cond_wait(&step_cv, &steps_mutex);
        }
        steps = steps_shared;
        steps_shared = 0;
        pthread_mutex_unlock(&steps_mutex);

        if (steps == -1) {
            break;
        }

        for (int z = 0; z < steps; ++z) {
            // TODO
        }
    }
    printf("exiting\n");
    pthread_exit(NULL);
}
