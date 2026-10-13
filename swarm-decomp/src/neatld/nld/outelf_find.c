#include "outelf.h"

static int outelf_find(struct outelf *oe, char *name, struct objent **po, Elf64_Sym **ps)
{
    int i, j;
    struct objent *o;
    Elf64_Sym *s;
    for (i = 0; i < oe->nobjs; i++) {
        o = &oe->objs[i];
        for (j = 0; j < o->nsyms; j++) {
            s = &o->syms[j];
            if (!ELF64_ST_BIND(s->st_info) || s->st_shndx == 0)
                continue;
            if (strcmp(name, o->strtab + s->st_name) == 0) {
                *po = o;
                *ps = s;
                return 0;
            }
        }
    }
    return 1;
}
