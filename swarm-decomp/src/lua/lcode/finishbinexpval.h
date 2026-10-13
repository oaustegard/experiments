#pragma once
#include "FuncState.h"
#include "expdesc.h"
#include "OpCode.h"
#include "TMS.h"

static void finishbinexpval(FuncState *fs, expdesc *e1, expdesc *e2, int op, int v2, int flip, int line, OpCode mmop, TMS event);
