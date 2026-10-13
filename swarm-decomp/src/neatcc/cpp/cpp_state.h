#pragma once
#include "buf.h"
/* more file-scope state of the cpp TU (static in the target) */
static int bufs_limit;
static int seen_macro;
static char seen_name[128];
static int elen;
static int ecur;
static char ebuf[1024];
static int hunk_len;
static int hunk_off;
static int nlocs;
static char *locs[1024];
