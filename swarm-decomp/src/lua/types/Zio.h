#pragma once
#include <stddef.h>
#include "lobject.h"
#include "lstate.h"

typedef struct Zio ZIO;

struct Zio {
  size_t n;
  const char *p;
  lua_Reader reader;
  void *data;
  lua_State *L;
};
