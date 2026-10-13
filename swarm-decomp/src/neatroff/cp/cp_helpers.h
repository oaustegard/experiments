#pragma once

char *quotednext(int (*next)(void), void (*back)(int));
int eval_up(char **s, int c);
int charread_delim(char **s, char *buf, char *cbuf);
void num_inc(int r, int inc);
char *num_str(int r);
char *num_getfmt(int r);
int ren_wid(int (*next)(void), void (*back)(int));
