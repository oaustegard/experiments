#pragma once
/* one ELF64 section header; 64 bytes */
struct elf_shdr {
	int name;	/* +0 */
	int type;	/* +4 */
	long flags;	/* +8 */
	long addr;	/* +16 */
	long offset;	/* +24 */
	long size;	/* +32 */
	int link;	/* +40 */
	int info;	/* +44 */
	long addralign;	/* +48 */
	long entsize;	/* +56 */
};
