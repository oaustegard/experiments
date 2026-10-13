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
    char outname[256] = "a.out";
    char libpath[256];
    struct stat st;
    unsigned char *objs[1024];
    unsigned char *obuf;
    char *a, *c, *p, *name;
    int i, j, k, n, nobj = 0, fd, shstr, cnt, t, off, acc, nsh;
    long d, pg, vd, ld, vb, fb, base, rs_off;
    unsigned long al;
    int sx, sy;
    Elf64_Rela *rp;
    Elf64_Shdr *sh;
    unsigned int v;

    if (argc <= 1)
        die("neatld: no object given!");
    memset(&oe, 0, sizeof oe);
    oe.eh.e_ident[0] = 0x7f;
    oe.eh.e_ident[1] = 'E';
    oe.eh.e_ident[2] = 'L';
    oe.eh.e_ident[3] = 'F';
    oe.eh.e_ident[4] = ELFCLASS64;
    oe.eh.e_ident[5] = ELFDATA2LSB;
    oe.eh.e_ident[6] = EV_CURRENT;
    oe.eh.e_type = ET_EXEC;
    oe.eh.e_version = EV_CURRENT;
    oe.eh.e_ehsize = sizeof oe.eh;
    oe.eh.e_phentsize = sizeof oe.ph[0];
    oe.eh.e_shentsize = sizeof oe.shdr[0];

    for (i = 1; i < argc; i++) {
        a = argv[i];
        if (a[0] != '-') {
            obuf = obj_add(&oe, a);
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
            objs[nobj++] = obuf;
            break;
        case 's':
            nosyms = 1;
            break;
        case 'p':
            secalign = 0x1000;
            break;
        case 'm':
            k = a[2] == 'd' ? 1 : (a[2] == 'b' ? 2 : 0);
            p = a[3] == '=' ? a + 4 : argv[++i];
            c = strchr(p, ':');
            v = hexnum(p);
            sec_vaddr[k] = v;
            sec_laddr[k] = c ? hexnum(c + 1) : v;
            sec_set[k] = 1;
            break;
        default:
            return main_usage();
        }
    }

    /* section layout: text segment */
    off = (0x11f + secalign) & -secalign;
    pg = (long)(off & 0xfffff000);
    off &= 0xfff;
    for (i = 0; i < oe.nsecs; i++) {
        struct secent *se = &oe.secs[i];
        if (se->shdr->sh_flags & SHF_EXECINSTR) {
            al = se->shdr->sh_addralign;
            if (al < 4)
                al = 4;
            off = (off + ((int)sec_vaddr[0] - 1) + al) & -al;
            off -= (int)sec_vaddr[0];
            base = off;
            off += se->shdr->sh_size;
            se->base = base + sec_vaddr[0];
            se->off = base + pg;
        }
    }
    oe.ph[0].p_type = PT_LOAD;
    oe.ph[0].p_flags = PF_R | PF_X;
    oe.ph[0].p_offset = pg;
    oe.ph[0].p_vaddr = sec_vaddr[0];
    oe.ph[0].p_paddr = sec_laddr[0];
    oe.ph[0].p_filesz = off;
    oe.ph[0].p_memsz = off;
    oe.ph[0].p_align = 0x1000;

    /* data segment */
    t = ((off + (secalign - 1) + (int)(pg & 0xfffff000)) & -secalign) - (int)(pg & 0xfffff000);
    pg += t;
    if (sec_set[1]) {
        vd = sec_vaddr[1] | (pg & 0xfff);
        ld = (pg & 0xfff) | sec_laddr[1];
    } else {
        vd = sec_vaddr[0] + t + 0x1000;
        ld = sec_laddr[0] + t + 0x1000;
    }
    acc = 0;
    for (i = 0; i < oe.nsecs; i++) {
        struct secent *se = &oe.secs[i];
        if (!(se->shdr->sh_flags & SHF_EXECINSTR) && se->shdr->sh_type != SHT_NOBITS) {
            base = acc;
            acc += se->shdr->sh_size;
            se->base = base + vd;
            se->off = base + pg;
        }
    }
    oe.ph[1].p_type = PT_LOAD;
    oe.ph[1].p_flags = PF_R | PF_W;
    oe.ph[1].p_offset = pg;
    oe.ph[1].p_vaddr = vd;
    oe.ph[1].p_paddr = ld;
    acc = (acc + 3) & -4;
    oe.ph[1].p_filesz = acc;
    oe.ph[1].p_memsz = acc;
    oe.ph[1].p_align = 0x1000;

    /* bss segment */
    off = ((secalign - 1 + (int)pg + acc) & -secalign) - (int)pg;
    fb = pg + off;
    if (sec_set[2]) {
        vb = (fb & 0xfff) | sec_vaddr[2];
    } else {
        vb = vd + off + 0x1000;
    }
    if (oe.nobjs > 0) {
        int comm = oe.comm_size;
        for (i = 0; i < oe.nobjs; i++) {
            o = &oe.objs[i];
            for (j = 0; j < o->nsyms; j++) {
                s = &o->syms[j];
                if (s->st_shndx == SHN_COMMON) {
                    n = oe.ncomms;
                    oe.ncomms = n + 1;
                    al = s->st_value;
                    if (al < 4)
                        al = 4;
                    comm = (comm + al - 1) & -al;
                    if (n > oe.comms_alloc) {
                        int cap = oe.comms_alloc ? oe.comms_alloc * 2 : 0x100;
                        oe.comms_alloc = cap;
                        oe.comms = realloc(oe.comms, cap * sizeof(struct commsym));
                        if (!oe.comms)
                            die("neatld: failed to allocate memory for BSS symbols");
                    }
                    oe.comms[n].off = comm;
                    comm += s->st_size;
                    oe.comms[n].sym = s;
                    oe.comm_size = comm;
                }
            }
        }
    }
    off = oe.comm_size;
    oe.comm_base = vb;
    for (i = 0; i < oe.nsecs; i++) {
        struct secent *se = &oe.secs[i];
        if (se->shdr->sh_type == SHT_NOBITS) {
            se->off = fb;
            al = se->shdr->sh_addralign;
            if (al < 4)
                al = 4;
            off = (off + ((int)vb - 1) + al) & -al;
            off -= (int)vb;
            base = off;
            off += se->shdr->sh_size;
            se->base = base + vb;
        }
    }
    oe.ph[2].p_type = PT_LOAD;
    oe.ph[2].p_flags = PF_R | PF_W;
    oe.ph[2].p_offset = fb;
    oe.ph[2].p_vaddr = vb;
    oe.ph[2].p_paddr = vb;
    oe.ph[2].p_filesz = 0;
    oe.ph[2].p_memsz = off;
    oe.ph[2].p_align = 0x1000;
    oe.nph = 3;

    /* apply relocations */
    for (i = 0; i < oe.nobjs; i++) {
        o = &oe.objs[i];
        for (j = 0; j < ((Elf64_Ehdr *)o->buf)->e_shnum; j++) {
            sh = &o->shdr[j];
            if (sh->sh_type != SHT_RELA)
                continue;
            cnt = sh->sh_size / sizeof(Elf64_Rela);
            rp = (Elf64_Rela *)(o->ehdr + sh->sh_offset);
            for (k = 0; k < cnt; k++, rp++) {
                Elf64_Shdr *ts = &o->shdr[sh->sh_info];
                t = (int)rp->r_info;
                s = &o->syms[ELF64_R_SYM(rp->r_info)];
                d = symval(&oe, o, s);
                switch (e_machine) {
                case 40:
                    t |= 0x10000;
                    break;
                case 62:
                    t |= 0x20000;
                    break;
                case 3:
                    t |= 0x40000;
                    break;
                default:
                    die("neatld: unknown relocation type!");
                }
                d += rp->r_addend;
                rs_off = (long)(o->ehdr + ts->sh_offset + rp->r_offset);
                switch (t) {
                default:
                    break;
                }
                (void)rs_off;
            }
        }
    }

    /* output file */
    oe.shoff = fb;
    fd = open(outname, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0)
        die("neatld: failed to create the output!");
    if (outelf_find(&oe, entry, &eo, &es))
        die_undef(entry);
    d = symval(&oe, eo, es);
    oe.eh.e_entry = d + sec_laddr[0] - sec_vaddr[0];
    if (!nosyms) {
        nsh = oe.nshdr;
        oe.nshdr = nsh + 6;
        shstr = outelf_str(&oe, "");
        oe.shdr[nsh + 4].sh_name = outelf_str(&oe, ".symtab");
        oe.shdr[nsh + 5].sh_name = outelf_str(&oe, ".strtab");
        oe.shdr[nsh + 1].sh_name = outelf_str(&oe, ".text");
        oe.shdr[nsh + 2].sh_name = outelf_str(&oe, ".data");
        oe.shdr[nsh + 3].sh_name = outelf_str(&oe, ".bss");
        os = outelf_sym(&oe);
        memset(os, 0, sizeof *os);
        for (i = 0; i < oe.nobjs; i++) {
            o = &oe.objs[i];
            for (j = 0; j < o->nsyms; j++) {
                s = &o->syms[j];
                name = o->strtab + s->st_name;
                if (*name && ELF64_ST_BIND(s->st_info) && s->st_shndx != SHN_UNDEF) {
                    os = outelf_sym(&oe);
                    os->st_name = outelf_str(&oe, name);
                    os->st_info = ELF64_ST_INFO(ELF64_ST_BIND(s->st_info), ELF64_ST_TYPE(s->st_info));
                    os->st_shndx = SHN_ABS;
                    os->st_value = symval(&oe, o, s);
                    os->st_size = s->st_size;
                }
            }
        }
        d = (int)oe.shoff;
        oe.eh.e_shoff = d;
        oe.eh.e_shnum = oe.nshdr;
        oe.shoff = d;
        sx = oe.nshdr * 64 + (int)d;
        oe.sysm_off = sx;
        sy = oe.nsysm * 24 + sx;
        oe.symstr_off = sy;
        oe.eh.e_shstrndx = nsh + 5;
        sh = &oe.shdr[nsh + 5];
        sh->sh_type = SHT_STRTAB;
        sh->sh_offset = oe.symstr_off;
        sh->sh_size = oe.symstr_len;
        sh = &oe.shdr[nsh + 4];
        sh->sh_type = SHT_SYMTAB;
        sh->sh_offset = oe.sysm_off;
        sh->sh_size = oe.nsysm * 24;
        sh->sh_link = nsh + 5;
        sh->sh_info = 0;
        sh->sh_entsize = 24;
        sh = &oe.shdr[nsh + 1];
        sh->sh_type = SHT_PROGBITS;
        sh->sh_flags = SHF_ALLOC | SHF_EXECINSTR;
        sh->sh_offset = oe.ph[0].p_offset;
        sh->sh_addr = oe.ph[0].p_vaddr;
        sh->sh_addralign = oe.ph[0].p_align;
        sh->sh_size = oe.ph[0].p_filesz;
        sh = &oe.shdr[nsh + 2];
        sh->sh_type = SHT_PROGBITS;
        sh->sh_flags = SHF_ALLOC | SHF_WRITE;
        sh->sh_offset = oe.ph[1].p_offset;
        sh->sh_addr = oe.ph[1].p_vaddr;
        sh->sh_addralign = oe.ph[1].p_align;
        sh->sh_size = oe.ph[1].p_filesz;
        sh = &oe.shdr[nsh + 3];
        sh->sh_type = SHT_NOBITS;
        sh->sh_flags = SHF_ALLOC | SHF_WRITE;
        sh->sh_offset = oe.ph[2].p_offset;
        sh->sh_addr = oe.ph[2].p_vaddr;
        sh->sh_addralign = oe.ph[2].p_align;
        sh->sh_size = oe.ph[2].p_filesz;
    }
    oe.eh.e_phoff = sizeof oe.eh;
    oe.eh.e_phnum = oe.nph;
    oe.eh.e_machine = e_machine;
    oe.eh.e_flags = e_flags;
    lseek(fd, 0, SEEK_SET);
    write(fd, &oe.eh, sizeof oe.eh);
    write(fd, oe.ph, oe.nph * sizeof(Elf64_Phdr));
    for (i = 0; i < oe.nsecs; i++) {
        sh = oe.secs[i].shdr;
        if (sh->sh_type != SHT_NOBITS) {
            lseek(fd, oe.secs[i].off, SEEK_SET);
            write(fd, oe.secs[i].obj->ehdr + sh->sh_offset, sh->sh_size);
        }
    }
    if (!nosyms) {
        lseek(fd, oe.shoff, SEEK_SET);
        write(fd, oe.shdr, oe.nshdr * sizeof(Elf64_Shdr));
        lseek(fd, oe.sysm_off, SEEK_SET);
        write(fd, oe.sysm, oe.nsysm * sizeof(Elf64_Sym));
        lseek(fd, oe.symstr_off, SEEK_SET);
        write(fd, oe.symstr, oe.symstr_len);
    }
    close(fd);
    for (i = 0; i < nobj; i++)
        free(objs[i]);
    free(oe.comms);
    free(oe.sysm);
    free(oe.symstr);
    return shstr;
}
