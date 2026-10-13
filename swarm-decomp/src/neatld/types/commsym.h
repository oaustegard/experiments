#pragma once
#include "sys_std.h"

/* common symbol -> offset; 16 bytes; array at 0x17c60 */
struct commsym {
    Elf64_Sym *sym;         /* +0 */
    int off;                /* +8 */
};
