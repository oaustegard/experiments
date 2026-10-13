#pragma once
#include "type.h"
/* one declarator being read by readname/globaldef; 0x120 bytes */
struct decl {
	char name[0x80];	/* +0: declarator name */
	char pre[0x80];		/* +0x80: prefix text (a name is used when pre is empty) */
	struct type t;		/* +0x100: the declared type */
	int pad;		/* +0x114 */
	int loc[2];		/* +0x118: location (a long), set by o_arg2loc or o_dsnew */
};
