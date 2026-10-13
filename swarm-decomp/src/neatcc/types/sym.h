#pragma once
/* one entry of the output symbol table; 24 bytes. Only str, type and
   shndx are known from put_sym; val and size are not yet inferred. */
struct sym {
	unsigned int str;	/* +0: offset into symstr */
	char type;	/* +4 */
	short shndx;	/* +6 */
	long val;	/* +8 */
	long size;	/* +16 */
};
