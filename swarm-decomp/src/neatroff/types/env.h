#pragma once
#include "wb.h"

struct env {
	int reg[64];		/* 0x000: number registers local to the environment */
	int tab[32];		/* 0x100: tab stops */
	char pad180[0x20];	/* 0x180 */
	void *fmt;		/* 0x1a0 */
	struct wb wb;		/* 0x1a8 */
	char tc[0x20];		/* 0x2218 */
	char lc[0x20];		/* 0x2238 */
	char hc[0x20];		/* 0x2258 */
	char mc[0x20];		/* 0x2278 */
};
