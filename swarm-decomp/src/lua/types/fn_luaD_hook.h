#pragma once
#include "lstate.h"

void luaD_hook (lua_State *L, int event, int line, int ftransfer, int ntransfer);
