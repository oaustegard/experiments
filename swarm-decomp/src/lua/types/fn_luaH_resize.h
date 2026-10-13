#pragma once
#include "lobject.h"
#include "lstate.h"

void luaH_resize (lua_State *L, Table *t, unsigned int nasize, unsigned int nhsize);
