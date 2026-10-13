#pragma once
#include <stddef.h>
#include "lobject.h"

typedef struct Mbuffer {
  char *buffer;
  size_t n;
  size_t buffsize;
} Mbuffer;
