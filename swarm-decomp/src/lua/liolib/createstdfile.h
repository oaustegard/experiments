#pragma once
#include "lauxlib.h"
#include <stdio.h>

static inline void createstdfile (lua_State *L, FILE *f, const char *k, const char *fname);
