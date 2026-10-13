#pragma once
#include "FuncState.h"
#include "expdesc.h"
#include "BinOpr.h"

void luaK_posfix (FuncState *fs, BinOpr opr, expdesc *e1, expdesc *e2, int line);
