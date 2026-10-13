#include "outelf.h"

static unsigned char *obj_add(struct outelf *oe, char *fname)
{
    struct stat st;
    int fd, size, len, n, i, j, k, count, off, nfound, want, def, base;
    unsigned char *buf, *hdr, *data;
    char *names;
    struct objent *o;

    fd = open(fname, 0);
    fstat(fd, &st);
    size = st.st_size;
    buf = malloc(size);
    read(fd, buf, size);
    close(fd);
    if (buf == NULL)
        die("neatld: cannot open object!");
    len = strlen(fname);
    if (len > 2 && fname[len - 2] == '.' && fname[len - 1] == 'a') {
        hdr = buf + 8;
        for (;;) {
            hdr[0x39] = 0;
            n = strtol((char *)hdr + 0x30, NULL, 10);
            n = (n + 1) & ~1;
            data = hdr + 0x3c;
            if (hdr[0] == '/') {
                if (hdr[1] == ' ') {
                    base = data - buf;
                    do {
                        nfound = 0;
                        count = __bswap_32(*(unsigned int *)(hdr + 0x3c));
                        names = (char *)hdr + 0x40 + count * 4;
                        for (i = 0; i < count; i++) {
                            want = 0;
                            def = 0;
                            for (k = 0; !def && k < oe->nobjs; k++) {
                                o = &oe->objs[k];
                                for (j = 0; j < o->nsyms; j++) {
                                    Elf64_Sym *s = &o->syms[j];
                                    if (ELF64_ST_BIND(s->st_info) && strcmp(names, o->strtab + s->st_name) == 0) {
                                        if (s->st_shndx != 0) {
                                            def = 1;
                                            break;
                                        }
                                        want = 1;
                                    }
                                }
                            }
                            if (!def && want) {
                                off = __bswap_32(*(unsigned int *)(hdr + 0x40 + i * 4));
                                outelf_add(oe, data + (off + 0x3c - base));
                                nfound++;
                            }
                            names += strlen(names) + 1;
                        }
                    } while (nfound);
                    return buf;
                }
                if (hdr[1] == '/' && hdr[2] == ' ')
                    outelf_add(oe, data);
            }
            hdr = data + n;
        }
    }
    outelf_add(oe, buf);
    return buf;
}
