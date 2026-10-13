#pragma once
#include "sbuf.h"

/* one diversion, 0x30 bytes; cdiv points at the current one, divs[] holds 16 */
struct div {
	struct sbuf buf;	/* 0x00: text collected by the diversion */
	int f10;		/* 0x10: name of the diversion (str_get/str_set key) */
	int f14;		/* 0x14 */
	int f18;		/* 0x18: set to -1 when the diversion is created */
	int f1c;		/* 0x1c */
	int f20;		/* 0x20: .mk-style register copies, see tr_di */
	int f24;		/* 0x24 */
	int f28;		/* 0x28 */
	int f2c;		/* 0x2c */
};
