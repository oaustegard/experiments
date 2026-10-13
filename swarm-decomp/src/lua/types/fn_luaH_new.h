#pragma once
#include <lua.h>
#include "lobject.h"
#include "lstate.h"
#include "Table.h"

Table *luaH_new(lua_State *L);
