#pragma once
#include <lua.h>

void *luaM_shrinkvector_(lua_State *L, void *block, int *size, int final_n, int size_elem);
