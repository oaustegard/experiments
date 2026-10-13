#pragma once
#include "fmt.h"

static int fmt_fillwords(struct fmt *f, int force);
static int fmt_safelines(void);
static long FMT_COST(int a, int b, int c, int d);
static int fmt_confchanged(struct fmt *f);
static void fmt_confupdate(struct fmt *f);
static int fmt_extractline(struct fmt *f, int n, int b);
