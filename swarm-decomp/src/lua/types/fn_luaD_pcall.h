#pragma once
#include "Pfunc.h"
#include "lstate.h"

int luaD_pcall(lua_State *L, Pfunc func, void *u, ptrdiff_t old_top, ptrdiff_t ef);
