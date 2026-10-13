#pragma once
#include "lstate.h"

#define LX_EXTRASPACE 8

typedef struct LX {
  unsigned char extra_[LX_EXTRASPACE];
  lua_State l;
} LX;
