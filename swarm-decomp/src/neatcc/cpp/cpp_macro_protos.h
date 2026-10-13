#pragma once
#include <ctype.h>
#include "cpp_protos.h"
#include "cpp_macros.h"
static void read_tilleol(char *d);
static int jumpcomment(void);
static int jumpstr(void);
static int jumpws(void);
static void read_word(char *d);
static void readarg(char *d);
