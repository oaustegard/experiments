#pragma once
#include "Udata.h"
#include "lstate.h"

Udata *luaS_newudata (lua_State *L, size_t s, int nuvalue);
