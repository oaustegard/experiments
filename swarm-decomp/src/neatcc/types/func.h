#pragma once
#include "type.h"
/* one entry of the funcs table; 0x131c bytes: the parameter types, the count
   and flags, the parameter names (32 x 128 bytes), then the function name */
struct func {
	struct type param[33];	/* +0 */
	int nargs;		/* +0x294 */
	int flags;		/* +0x298 */
	char arg[32][0x80];	/* +0x29c */
	char name[0x80];	/* +0x129c */
};
