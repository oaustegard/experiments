#pragma once
#include <lua.h>
#include "Table.h"

void luaH_setint(lua_State *L, Table *t, lua_Integer key, TValue *value);
