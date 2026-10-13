#include <ctype.h>
#include "tr_args.h"
#include "ext.h"
#include "libc.h"

static void tr_ad(struct tr_args *a)
{
	char *s = a->arg[0];
	int c;

	*nreg(map(".na")) = 0;
	if (!s)
		return;
	if (isdigit((unsigned char)*s)) {
		*nreg(0x6a) = strtol(s, 0, 10) & 0xfff;
		return;
	}
	*nreg(0x6a) = 0;
	while ((c = *s++) != 0) {
		switch (c) {
		case 'c':
			*nreg(0x6a) |= 1;
			break;
		case 'r':
			*nreg(0x6a) |= 4;
			break;
		case 'q':
			*nreg(0x6a) |= 0x80;
			break;
		case 'p':
			*nreg(0x6a) |= 0x20;
			break;
		case 'l':
			*nreg(0x6a) |= 2;
			break;
		case 'k':
			*nreg(0x6a) |= 0x40;
			break;
		case 'b':
			*nreg(0x6a) |= 0x10;
			break;
		case 'R':
			*nreg(0x6a) |= 0x200;
			break;
		case 'L':
			*nreg(0x6a) |= 0x100;
			break;
		case 'C':
			*nreg(0x6a) |= 0x400;
			break;
		}
	}
	if ((*nreg(0x6a) & 0x20) && !(*nreg(0x6a) & 7))
		*nreg(0x6a) |= 0x10;
}
