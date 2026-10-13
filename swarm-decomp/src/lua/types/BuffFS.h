#pragma once
#include "lstate.h"

#define BUFVFS 200

typedef struct BuffFS {
  lua_State *L;
  int pushed;
  int nb;
  char buff[BUFVFS];
} BuffFS;
