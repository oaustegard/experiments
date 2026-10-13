#include <string.h>

static int
lookupcmp(const void *pa, const void *pb)
{
	const struct otf_feat *a = pa, *b = pb;
	int r;

	r = strcmp(a->name, b->name);
	if (r)
		return r;
	if (mkfn_featrank(a->name, a->tag) != mkfn_featrank(a->name, b->tag))
		return mkfn_featrank(a->name, a->tag) - mkfn_featrank(a->name, b->tag);
	return a->idx - b->idx;
}
