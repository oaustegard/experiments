#include "pref.h"
#include "fn_mextend.h"
#include "iset_put.h"

void pref_put(struct pref *p, char *s, int v)
{
	unsigned int h;
	int i;
	void *set;

	if (p->cnt >= p->cap) {
		int n = p->cnt + 0x400;
		p->key = mextend(p->key, p->cap, n, sizeof(char *));
		p->val = mextend(p->val, p->cap, n, sizeof(int));
		p->cap = n;
	}
	set = p->set;
	i = p->cnt++;
	p->key[i] = s;
	p->val[i] = v;
	h = (unsigned char)s[0] << 5 | (unsigned char)s[1];
	iset_put(set, h % 1023, i);
}
