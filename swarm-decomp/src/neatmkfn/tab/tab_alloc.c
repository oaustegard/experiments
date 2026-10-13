#include <stdlib.h>
#include "tab.h"

struct tab *tab_alloc(int n)
{
	struct tab *t;
	int i;

	t = calloc(sizeof(struct tab), 1);
	t->key = malloc(n * sizeof(char *));
	t->val = malloc(n * sizeof(void *));
	t->idx = malloc(n * sizeof(int));
	for (i = 0; i < 256; i++)
		t->hash[i] = -1;
	return t;
}
