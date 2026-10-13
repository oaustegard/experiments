#pragma once
#include <float.h>
#include "lua_api.h"
#include "lstrlib_macros.h"

#define L_FMTFLAGSF "-+#0 "
#define L_FMTFLAGSX "-#0"
#define L_FMTFLAGSI "-+0 "
#define L_FMTFLAGSU "-0"
#define L_FMTFLAGSC "-"
#define MAX_FORMAT 32
#define MAX_ITEMF (110 + DBL_MAX_10_EXP)
#define MAX_ITEM 120
