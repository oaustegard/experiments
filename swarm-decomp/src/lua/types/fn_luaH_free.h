#pragma once
#include <lua.h>
#include "Table.h"
#include "lstate.h"

void luaH_free(lua_State *L, Table *t);
