#pragma once
#include "LClosure.h"
#include "lstate.h"

void luaF_freeproto (lua_State *L, Proto *f);
