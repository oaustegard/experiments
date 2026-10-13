#pragma once
#include "buf.h"
/* file-scope state of the cpp TU (static in the target) */
static char *buf;
static long cur;
static long len;
static int bufs_n;
static struct buf bufs[32];
static int enext;
