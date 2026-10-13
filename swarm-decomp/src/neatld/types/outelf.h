#pragma once
#include "objent.h"
#include "secent.h"
#include "commsym.h"

struct outelf {
    char pad0[0xe048];
    struct secent secs[1024];       /* 0xe048 */
    int nsecs;                      /* 0x16048 */
    char pad1[4];
    struct objent objs[128];        /* 0x16050 */
    int nobjs;                      /* 0x17c50 */
    char pad2[0xc];
    struct commsym *comms;          /* 0x17c60 */
    int ncomms;                     /* 0x17c68 */
    char pad3[4];
    long comm_base;                 /* 0x17c70 */
    char pad4[0x27c88 - 0x17c78];
    char *symstr;                   /* 0x27c88 */
    int symstr_len;                 /* 0x27c90 */
    int symstr_alloc;               /* 0x27c94 */
    Elf64_Sym *sysm;                /* 0x27c98 */
    int nsysm;                      /* 0x27ca0 */
    int sysm_alloc;                 /* 0x27ca4 */
};
