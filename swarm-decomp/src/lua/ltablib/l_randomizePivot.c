#include <time.h>
#include <string.h>

static unsigned int l_randomizePivot (void) {
  clock_t c = clock();
  time_t t = time(NULL);
  unsigned int buff[sizeof(c) / sizeof(unsigned int) + sizeof(t) / sizeof(unsigned int)];
  unsigned int i, rnd = 0;
  memcpy(buff, &c, sizeof(c));
  memcpy(buff + sizeof(c) / sizeof(unsigned int), &t, sizeof(t));
  for (i = 0; i < sizeof(buff) / sizeof(buff[0]); i++)
    rnd += buff[i];
  return rnd;
}
