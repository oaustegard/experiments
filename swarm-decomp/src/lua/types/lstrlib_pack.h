#pragma once
#include <limits.h>
#include "lua_api.h"
#include "Header.h"
#include "KOption.h"

#define NB CHAR_BIT
#define MC ((1 << NB) - 1)
#define SZINT ((int)sizeof(lua_Integer))
#define LUAL_PACKPADBYTE 0x00
