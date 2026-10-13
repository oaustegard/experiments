#pragma once
#include <lua.h>
#include "LClosure.h"

LClosure *luaF_newLclosure (lua_State *L, int nupvals);
