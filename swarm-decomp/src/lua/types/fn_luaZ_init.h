#pragma once
#include "lstate.h"
#include "Zio.h"

void luaZ_init (lua_State *L, ZIO *z, lua_Reader reader, void *data);
