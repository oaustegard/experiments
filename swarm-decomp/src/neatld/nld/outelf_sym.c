#include "outelf.h"

static Elf64_Sym *outelf_sym(struct outelf *oe)
{
    if (oe->nsysm >= oe->sysm_alloc) {
        int a = oe->sysm_alloc ? oe->sysm_alloc * 2 : 0x100;
        oe->sysm_alloc = a;
        oe->sysm = realloc(oe->sysm, a * sizeof(Elf64_Sym));
        if (!oe->sysm)
            die("neatld: failed to allocate oe->sysm");
    }
    return &oe->sysm[oe->nsysm++];
}
