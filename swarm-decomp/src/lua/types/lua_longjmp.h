#pragma once
#include <setjmp.h>
#include "lstate.h"

struct lua_longjmp {
  struct lua_longjmp *previous;
  jmp_buf b;
  volatile int status;
};
