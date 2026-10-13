#pragma once
#include "FuncState.h"
#include "expdesc.h"
#include "BinOpr.h"

void luaK_infix (FuncState *fs, BinOpr op, expdesc *v);
