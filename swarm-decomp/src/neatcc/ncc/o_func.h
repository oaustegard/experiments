#pragma once
void o_func_beg(char *name, int nargs, int isglobal, int flags);
void o_func_end(void);
long o_arg2loc(int i);
long o_dsnew(char *name, long size, int isglobal);
