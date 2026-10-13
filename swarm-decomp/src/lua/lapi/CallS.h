#pragma once
#include "lstate.h"

/* argument block of lapi.c's f_call / lua_pcallk */
struct CallS {
  StkId func;
  int nresults;
};
