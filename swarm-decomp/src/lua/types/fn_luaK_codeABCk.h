#pragma once
#include "FuncState.h"
#include "OpCode.h"

int luaK_codeABCk(FuncState *fs, OpCode o, int a, int b, int c, int k);
