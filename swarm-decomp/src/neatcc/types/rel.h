#pragma once
/* one relocation entry; 24 bytes, laid out like Elf64_Rela. out_rel
   sets off and info; addend is not written by out_rel. */
struct rel {
	long off;	/* +0: r_offset */
	long info;	/* +8: symbol index << 32 | relocation type */
	long addend;	/* +16 */
};
