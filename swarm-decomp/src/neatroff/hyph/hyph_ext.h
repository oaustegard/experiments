#pragma once
#include "dict.h"
#include "pref.h"

/* functions of other translation units of neatroff that the hyph TU calls.
   Kept TU-local: dict/ and tr/ declare some of them with other signatures. */
int dict_get(struct dict *d, char *s);
void dict_put(struct dict *d, char *s, int v);
struct dict *dict_make(int n, int f);
void *pref_make(void);
void pref_put(struct pref *p, char *s, int v);
void pref_free(struct pref *p);
int utf8one(const char *s);
int utf8read(char **s, char *out);
int pref_prefix(struct pref *p, char *s, int *len);
