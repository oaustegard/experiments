#include "trcmd.h"
#include "ext.h"
#include "ext.h"
#include "ext.h"

void tr_init(void)
{
	struct trcmd *p;

	for (p = cmds; p != cmds + 118; p++)
		str_dset(map(p->name), p);
	cmap = dict_make(1024, 0);
}
