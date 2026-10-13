#include "pref.h"
#include "fn_xmalloc.h"
#include "fn_mextend.h"
#include "fn_iset_make.h"

struct pref *pref_make(void)
{
	struct pref *p = xmalloc(sizeof(struct pref));
	*p = (struct pref){ .cnt = 1 };
	p->set = iset_make();
	p->key = mextend(p->key, p->cap, 0x400, sizeof(char *));
	p->val = mextend(p->val, p->cap, 0x400, sizeof(int));
	p->cap = 0x400;
	return p;
}
