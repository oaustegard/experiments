#pragma once
/* prototypes of the static functions of the cpp TU that are called before their definition */
static int include_file(char *name);
static int buf_arg_find(char *name);
static long evalbitor(void);
static long evaleq(void);
static long cpp_eval(void);
static int cpp_cmd(void);
static void jumpifs(int taken);
static void macro_expand(char *name);
