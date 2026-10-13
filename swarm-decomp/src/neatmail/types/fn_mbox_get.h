#pragma once
#include "mbox.h"

int mbox_get(struct mbox *m, int i, char **s, long *n);
