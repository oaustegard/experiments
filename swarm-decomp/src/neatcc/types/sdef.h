#pragma once
#include "decl.h"
/* one entry of the structs table; 0x9090 bytes: the tag, 128 field
   declarations, then the field count */
struct sdef {
	char name[0x80];		/* +0 */
	struct decl fld[128];		/* +0x80 */
	int nfld;			/* +0x9080 */
	int pad[3];			/* +0x9084 */
};
