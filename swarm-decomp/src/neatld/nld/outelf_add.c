#include "outelf.h"

static void outelf_add(struct outelf *oe, unsigned char *buf)
{
    Elf64_Ehdr *eh = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sh = (Elf64_Shdr *)(buf + eh->e_shoff);
    Elf64_Shdr *s;
    struct objent *o;
    int n, i;

    if (eh->e_type != ET_REL)
        return;
    e_machine = eh->e_machine;
    e_flags = eh->e_flags;
    n = oe->nobjs;
    if (n > 0x7f)
        die("neatld: MAXOBJS reached!");
    oe->nobjs = n + 1;
    o = &oe->objs[n];
    o->ehdr = buf;
    o->buf = buf;
    o->shdr = sh;
    o->shstrtab = (char *)buf + sh[eh->e_shstrndx].sh_offset;
    for (i = 0; i < eh->e_shnum; i++) {
        s = &sh[i];
        if (s->sh_type != SHT_SYMTAB)
            continue;
        o->strtab = (char *)buf + sh[s->sh_link].sh_offset;
        o->syms = (Elf64_Sym *)(buf + s->sh_offset);
        o->nsyms = s->sh_size / sizeof(Elf64_Sym);
    }
    for (i = 0; i < eh->e_shnum; i++) {
        s = &sh[i];
        if (!(s->sh_flags & 7))
            continue;
        n = oe->nsecs;
        if (n > 0x3ff)
            die("neatld: MAXSECS reached!");
        oe->nsecs = n + 1;
        oe->secs[n].shdr = s;
        oe->secs[n].obj = o;
    }
}
