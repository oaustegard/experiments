#pragma once
/* one declarator being read by readname/globaldef; 0x120 bytes, name at +0x100 */
struct decl {
	char pre[0x100];
	char name[0x20];
};
