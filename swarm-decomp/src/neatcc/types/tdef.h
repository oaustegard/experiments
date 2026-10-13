#pragma once
#include "type.h"
/* one entry of the typedefs table; 0x94 bytes: the name, then the type */
struct tdef {
	char name[0x80];	/* +0 */
	struct type t;		/* +0x80 */
};
