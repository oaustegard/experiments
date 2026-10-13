#pragma once
#include <stddef.h>
#include "lstate.h"

void luaM_free_(lua_State *L, void *block, size_t osize);
