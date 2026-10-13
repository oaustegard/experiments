#pragma once
#include "lstate.h"

void luaC_checkfinalizer (lua_State *L, GCObject *o, Table *mt);
