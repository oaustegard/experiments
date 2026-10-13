#pragma once
#include "sbuf.h"

struct wb {
	struct sbuf buf;	/* 0x00 */
	union {
		int a10[8];	/* 0x10: set to -1 by wb_reset */
		struct {
			int f10[4];	/* 0x10: set to -1 by wb_init */
			int f20[4];	/* 0x20: set by wb_fnszset */
		};
	};
	int part;		/* 0x30 */
	int cost;		/* 0x34 */
	int a38[2];		/* 0x38 */
	int hpos;		/* 0x40 */
	int vpos;		/* 0x44 */
	int a48[3];		/* 0x48 */
	int a54[4];		/* 0x54: constant set by wb_reset */
	char pad64[0x2068 - 0x64];	/* 0x64: italic flag and such */
	int f2068;		/* 0x2068 */
	int f206c;		/* 0x206c */
};
