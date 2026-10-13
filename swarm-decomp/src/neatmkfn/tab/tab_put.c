#include "tab.h"

void tab_put(struct tab *t, const char *key, const void *val)
{
	t->key[t->n] = (char *)key;
	t->val[t->n] = (void *)val;
	t->idx[t->n] = t->hash[(unsigned char)*key];
	t->hash[(unsigned char)*key] = t->n;
	t->n++;
}
