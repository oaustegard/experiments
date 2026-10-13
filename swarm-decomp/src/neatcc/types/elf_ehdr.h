#pragma once
/* the ELF64 file header; 64 bytes */
struct elf_ehdr {
	int magic;		/* +0: 0x464c457f */
	short ident_class;	/* +4: EI_CLASS, EI_DATA */
	char ident_version;	/* +6: EI_VERSION */
	char ident_pad[9];	/* +7 */
	short type;		/* +16 */
	short machine;		/* +18 */
	int version;		/* +20 */
	long entry;		/* +24 */
	long phoff;		/* +32 */
	long shoff;		/* +40 */
	int flags;		/* +48 */
	short ehsize;		/* +52 */
	short phentsize;	/* +54 */
	short phnum;		/* +56 */
	short shentsize;	/* +58 */
	short shnum;		/* +60 */
	short shstrndx;		/* +62 */
};
