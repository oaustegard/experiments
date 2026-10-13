#pragma once
#include <lua.h>
#include "GCObject.h"

void luaC_barrier_(lua_State *L, GCObject *o, GCObject *v);
