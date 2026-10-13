#pragma once
#include "lua.h"

static const char *findfile (lua_State *L, const char *name, const char *pname,
                             const char *dirsep);
