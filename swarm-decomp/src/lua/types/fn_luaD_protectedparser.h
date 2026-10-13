#pragma once
#include "lstate.h"
#include "Zio.h"

int luaD_protectedparser (lua_State *L, ZIO *z, const char *name, const char *mode);
