#pragma once
#include "lua.h"

static void setpath (lua_State *L, const char *fieldname, const char *envname,
                                   const char *dft);
