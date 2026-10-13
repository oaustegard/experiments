#pragma once
#include "lobject.h"
#include "lstate.h"

void *luaM_saferealloc_ (lua_State *L, void *block, size_t osize, size_t nsize);
