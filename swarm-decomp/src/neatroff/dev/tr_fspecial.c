#include "tr_args.h"
#include "fn_dev_pos.h"

void tr_fspecial(struct tr_args *a)
{
	int n, k, cnt, i;
	long row;
	char **p;

	if (a->arg[0] == NULL)
		return;
	n = dev_pos(a->arg[0]);
	if (n <= 0)
		return;
	row = n;
	cnt = 0;
	for (p = &a->arg[1]; p != (char **)((char *)a + 0x100); p++) {
		i = cnt;
		if (*p == NULL)
			break;
		k = dev_pos(*p);
		if (k > 0 && (unsigned)cnt <= 3) {
			fn_special[n][cnt++] = k;
			i = cnt;
		}
	}
	if ((unsigned)i <= 3)
		fn_special[n][cnt] = 0;
}
