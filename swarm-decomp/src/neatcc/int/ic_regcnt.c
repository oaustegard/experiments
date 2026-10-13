#include "ic_ent.h"

int ic_regcnt(struct ic_ent *e)
{
	unsigned long op = e->op;

	if (op & 0xf0)
		return !(op & 0x700000) + 1;
	if (op & 0x100)
		return !(op & 0x700000);
	if (op & 0x200)
		return !(op & 0x700000);
	if (op & 0x400)
		return !(op & 0x700000);
	if (op & 0x800)
		return 3;
	if (op & 0x1000)
		return 0;
	if (op & 0x2000)
		return 1;
	if (op & 0x4000)
		return !(op & 0x700000) + 1;
	if (op & 0x8000)
		return 1;
	if (op & 0x10000)
		return (op >> 20) & 1;
	if (op & 0x20000)
		return ((op & 0x100000) != 0) + 1;
	return 0;
}
