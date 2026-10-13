#pragma once
#include <lua.h>

void *luaM_growaux_(lua_State *L, void *block, int nelems, int *size, int size_elem, int limit, const char *what);
