#pragma once
#include "lobject.h"
#include "lstate.h"

void luaC_fix(lua_State *L, GCObject *o);
