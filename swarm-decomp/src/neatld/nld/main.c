#include "outelf.h"

static int main_usage(void)
{
    puts("Usage: neatld [options] objects\n");
    puts("Options:");
    puts("  -o out          set the output file");
    puts("  -l lib          link with library lib");
    puts("  -L dir          search dir for libraries");
    puts("  -s              do not include a symbol table");
    puts("  -mXvaddr:laddr  section virtual/load address");
    puts("  -p              page-align sections");
    puts("  -e              entry point symbol");
    return 1;
}

int main(int argc, char **argv)
{
    struct outelf oe;
    struct objent *eo, *o;
    Elf64_Sym *es, *os, *s;
    char outname[0xf0];
    char libpath[256];
    struct stat st;
    Elf64_Ehdr eh;
    unsigned char *objs[64];
    unsigned char *obuf;
    char *a, *c;
    int i, j, k, nobj = 0, fd, n, shstr;
    long d;

    if (argc <= 1)
        die("neatld: no object given!");
    memset(&oe, 0, sizeof oe);
    strcpy(outname, "a.out");
    for (i = 1; i < argc; i++) {
        a = argv[i];
        if (a[0] != '-') {
            obuf = obj_add(&oe, a);
            if (nobj < 64)
                objs[nobj++] = obuf;
            continue;
        }
        switch (a[1]) {
        case 'o':
            strcpy(outname, a[2] ? a + 2 : argv[++i]);
            break;
        case 'e':
            entry = a[2] ? a + 2 : argv[++i];
            break;
        case 'L':
            libdirs[nlibdirs++] = a[2] ? a + 2 : argv[++i];
            break;
        case 'l':
            for (n = 0; n < nlibdirs; n++) {
                sprintf(libpath, "%s/lib%s.a", libdirs[n], a + 2);
                if (stat(libpath, &st) == 0)
                    break;
            }
            if (n >= nlibdirs)
                die("neatld: cannot find library!");
            obuf = obj_add(&oe, libpath);
            if (nobj < 64)
                objs[nobj++] = obuf;
            break;
        case 's':
            nosyms = 1;
            break;
        case 'p':
            secalign = 0x1000;
            break;
        case 'm':
            k = a[2] == 'd' ? 1 : 0;
            c = strchr(a + 3, ':');
            sec_vaddr[k] = hexnum(a + 3);
            if (c)
                sec_laddr[k] = hexnum(c + 1);
            sec_set[k] = 1;
            break;
        default:
            return main_usage();
        }
    }
    if (nobj == 0)
        die("neatld: no object given!");
    if (outelf_find(&oe, entry, &eo, &es))
        die_undef(entry);
    d = symval(&oe, eo, es);
    sec_vaddr[0] = d;
    shstr = outelf_str(&oe, "");
    outelf_str(&oe, ".symtab");
    outelf_str(&oe, ".strtab");
    outelf_str(&oe, ".text");
    outelf_str(&oe, ".data");
    outelf_str(&oe, ".bss");
    if (!nosyms) {
        for (i = 0; i < oe.nobjs; i++) {
            o = &oe.objs[i];
            for (j = 0; j < o->nsyms; j++) {
                s = &o->syms[j];
                if (ELF64_ST_BIND(s->st_info) && s->st_shndx != 0 && s->st_name) {
                    os = outelf_sym(&oe);
                    os->st_name = outelf_str(&oe, o->strtab + s->st_name);
                    os->st_info = s->st_info;
                    os->st_shndx = s->st_shndx;
                    os->st_value = symval(&oe, o, s);
                }
            }
        }
    }
    for (i = 0; i < oe.nsecs; i++) {
        os = outelf_sym(&oe);
        os->st_name = shstr;
        os->st_info = ELF64_ST_INFO(STB_LOCAL, STT_SECTION);
        os->st_shndx = i;
    }
    fd = open(outname, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0)
        die("neatld: failed to create the output!");
    memset(&eh, 0, sizeof eh);
    eh.e_type = ET_EXEC;
    eh.e_machine = e_machine;
    eh.e_flags = e_flags;
    eh.e_entry = d;
    eh.e_ehsize = sizeof eh;
    lseek(fd, 0, SEEK_SET);
    write(fd, &eh, sizeof eh);
    for (i = 0; i < oe.nsecs; i++) {
        Elf64_Shdr *sp = oe.secs[i].shdr;
        if (sp->sh_type != SHT_NOBITS) {
            lseek(fd, oe.secs[i].base, SEEK_SET);
            write(fd, oe.secs[i].obj->buf + sp->sh_offset, sp->sh_size);
        }
    }
    if (!nosyms) {
        lseek(fd, oe.sysm_alloc, SEEK_SET);
        write(fd, oe.sysm, oe.nsysm * sizeof(Elf64_Sym));
        lseek(fd, oe.symstr_alloc, SEEK_SET);
        write(fd, oe.symstr, oe.symstr_len);
    }
    for (i = 0; i < nobj; i++)
        free(objs[i]);
    close(fd);
    return shstr;
}
