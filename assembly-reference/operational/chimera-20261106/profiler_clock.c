#include <time.h>
double assembly_profile_time(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + 1e-9*(double)now.tv_nsec;
}
