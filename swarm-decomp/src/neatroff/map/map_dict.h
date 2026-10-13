#pragma once

#include "dict.h"

/* dict/ API as used by this TU only: kept local because other TUs declare
   dict_put/dict_make with void * parameters */
struct dict *dict_make(int n, int f);
void dict_put(struct dict *d, char *s, int v);
int dict_idx(struct dict *d, char *s);
char *dict_key(struct dict *d, int i);
