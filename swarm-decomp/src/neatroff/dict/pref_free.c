#include <stdlib.h>
#include "pref.h"
#include "fn_iset_free.h"

void pref_free(struct pref *p)
{
	free(p->val);
	free(p->key);
	iset_free(p->set);
	free(p);
}
