#pragma once
#include "sys_std.h"

/* one input object file; 56 bytes; array oe->objs[128] at 0x16050 */
struct objent {
    unsigned char *ehdr;    /* +0x00 file image */
    unsigned char *buf;     /* +0x08 same pointer */
    Elf64_Shdr *shdr;       /* +0x10 section header table */
    Elf64_Sym *syms;        /* +0x18 symbol table */
    int nsyms;              /* +0x20 */
    char *strtab;           /* +0x28 */
    char *shstrtab;         /* +0x30 */
};
