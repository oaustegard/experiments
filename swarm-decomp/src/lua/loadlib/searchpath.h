#pragma once
#include "lua.h"

static const char *searchpath (lua_State *L, const char *name,
                               const char *path, const char *sep,
                               const char *dirsep);
