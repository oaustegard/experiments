#pragma once
#include "sys.h"
#include "buf.h"
#include "cpp_vars.h"
static void buf_new(int kind, char *s, long n);
static void buf_pop(void);
void die(char *fmt, ...);
static void macro_define(void);
static int eval_tok(void);
static long evalor(void);
static long evalmul(void);
static long evalcmp(void);
