#include <string.h>
#include "pref.h"
#include "iset_get.h"

int pref_prefix(struct pref *p, char *s, int *len)
{
	unsigned int h;
	int *r;
	int i;
	char *k;
	int n;

	h = (unsigned char)s[0] << 5 | (unsigned char)s[1];
	r = iset_get(p->set, h % 1023);
	while (r != NULL && (i = r[++*len]) >= 0) {
		k = p->key[i];
		n = strlen(k);
		if (strncmp(k, s, n) == 0)
			return p->val[i];
	}
	return -1;
}
