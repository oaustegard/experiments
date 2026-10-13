#pragma once
#include <lua.h>
#include "GCObject.h"

void luaC_barrierback_(lua_State *L, GCObject *o);
