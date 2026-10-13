#pragma once

/* externs used by the font TU only; other TUs declare dict/iset/utf8 with void * */
struct dict *dict_make(int n, int dup);
int dict_get(struct dict *d, char *key);
void iset_put(struct iset *s, int a, int b);
int utf8read(char **src, char *dst);
