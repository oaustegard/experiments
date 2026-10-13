#pragma once
#include "src.h"

/* file-scope state of the input stack (is_stub() ignores brace-less .c files) */
static struct src *buf;
static int nfiles;
static char files[16][1024];
static int cfile;
