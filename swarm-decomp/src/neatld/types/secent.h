#pragma once
#include "sys_std.h"

/* one loaded section; 32 bytes; array oe->secs[1024] at 0xe048 */
struct secent {
    Elf64_Shdr *shdr;       /* +0x00 */
    struct objent *obj;     /* +0x08 */
    long base;              /* +0x10 output address */
    long pad;               /* +0x18 */
};
