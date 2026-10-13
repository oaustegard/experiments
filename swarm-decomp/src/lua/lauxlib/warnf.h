#pragma once
#include "lua.h"

static void warnfoff(void *ud, const char *message, int tocont);
static void warnfon(void *ud, const char *message, int tocont);
static void warnfcont(void *ud, const char *message, int tocont);
