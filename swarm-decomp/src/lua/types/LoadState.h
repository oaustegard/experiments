#pragma once
#include "lstate.h"
#include "Zio.h"

typedef struct LoadState {
  lua_State *L;
  ZIO *Z;
  const char *name;
} LoadState;
