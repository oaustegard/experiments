#pragma once
#include "objent.h"
#include "secent.h"
#include "commsym.h"

/* whole output image state; sizeof == 0x27cc0; a local in main() */
struct outelf {
    Elf64_Ehdr eh;                  /* 0x00000 output ELF header */
    Elf64_Phdr ph[1024];            /* 0x00040 program headers */
    int nph;                        /* 0x0e040 */
    struct secent secs[1024];       /* 0x0e048 */
    int nsecs;                      /* 0x16048 */
    struct objent objs[128];        /* 0x16050 */
    int nobjs;                      /* 0x17c50 */
    char pad2[0xc];
    struct commsym *comms;          /* 0x17c60 */
    int ncomms;                     /* 0x17c68 */
    int comms_alloc;                /* 0x17c6c */
    long comm_base;                 /* 0x17c70 */
    int comm_size;                  /* 0x17c78 */
    Elf64_Shdr shdr[1024];          /* 0x17c80 output section headers */
    int nshdr;                      /* 0x27c80 */
    char *symstr;                   /* 0x27c88 */
    int symstr_len;                 /* 0x27c90 */
    int symstr_alloc;               /* 0x27c94 */
    Elf64_Sym *sysm;                /* 0x27c98 */
    int nsysm;                      /* 0x27ca0 */
    int sysm_alloc;                 /* 0x27ca4 */
    long shoff;                     /* 0x27ca8 file offset of section headers */
    long sysm_off;                  /* 0x27cb0 file offset of symbols */
    long symstr_off;                /* 0x27cb8 file offset of symbol strings */
};
