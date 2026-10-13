#include "outelf.h"

static long symval(struct outelf *oe, struct objent *obj, Elf64_Sym *sym)
{
    int type = ELF64_ST_TYPE(sym->st_info);
    Elf64_Shdr *shp;
    char *name = NULL;
    int i;
    Elf64_Addr val;

    if (obj)
        name = obj->strtab + sym->st_name;
    if (type <= STT_FUNC) {
        if (name && *name && sym->st_shndx == SHN_UNDEF) {
            if (outelf_find(oe, name, &obj, &sym))
                die_undef(name);
        }
        if (sym->st_shndx == SHN_COMMON) {
            for (i = 0; i < oe->ncomms; i++)
                if (oe->comms[i].sym == sym)
                    return oe->comm_base + oe->comms[i].off;
            return 0;
        }
        val = sym->st_value;
        shp = &obj->shdr[sym->st_shndx];
        for (i = 0; i < oe->nsecs; i++)
            if (oe->secs[i].shdr == shp)
                return (int)val + oe->secs[i].base;
        return 0;
    }
        if (type != STT_SECTION)
            return 0;
        shp = &obj->shdr[sym->st_shndx];
        for (i = 0; i < oe->nsecs; i++)
            if (oe->secs[i].shdr == shp)
                return oe->secs[i].base;
        return 0;
}
