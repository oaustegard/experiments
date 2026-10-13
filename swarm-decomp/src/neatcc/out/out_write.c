#include <stdlib.h>
#include "fn_out_write.h"
#include "symtab.h"
#include "reltab.h"
#include "elftab.h"
#include "strtab.h"
#include "symstr_add.h"

void
out_write(int fd, char *cs, long csn, char *ds, long dsn)
{
	long *map;
	int i, last;
	long off, bss;
	struct sym *s;
	struct sym t;

	shdr[1].name = symstr_add(".cs");
	shdr[2].name = symstr_add(".rela.cs");
	shdr[5].name = symstr_add(".ds");
	shdr[6].name = symstr_add(".rela.ds");
	ehdr.ident_class = 0x102;
	off = 0x240 + csn;
	ehdr.type = 1;
	ehdr.machine = 0x3e;
	ehdr.version = 1;
	ehdr.magic = 0x464c457f;
	ehdr.ident_version = 1;
	ehdr.shoff = 0x40;
	ehdr.ehsize = 0x40;
	ehdr.shentsize = 0x40;
	ehdr.shnum = 8;
	ehdr.shstrndx = 4;
	shdr[1].type = 1;
	shdr[1].flags = 6;
	shdr[1].offset = 0x240;
	shdr[1].size = csn;
	shdr[2].type = 4;
	shdr[2].entsize = 0x18;
	shdr[2].offset = off;
	shdr[2].size = 24 * csrel_n;
	off += 24 * csrel_n;
	shdr[3].type = 2;
	shdr[3].offset = off;
	shdr[3].size = 24 * syms_n;
	shdr[3].entsize = 0x18;
	shdr[3].link = 4;
	shdr[2].link = 3;
	shdr[2].info = 1;
	shdr[1].addralign = 0x10;
	shdr[1].entsize = 1;

	map = malloc(8 * syms_n);
	for (i = 0; i < syms_n; i++)
		map[i] = i;
	last = syms_n - 1;
	i = 1;
	for (;;) {
		if (i < last && (syms[i].type >> 4) == 0) {
			i++;
			continue;
		}
		if (i > last)
			break;
		while (last >= i && (syms[last].type >> 4) != 0)
			last--;
		if (last <= i)
			break;
		t = syms[last];
		syms[last] = syms[i];
		syms[i] = t;
		map[i] = last;
		map[last] = i;
	}
	for (i = 0; i < csrel_n; i++) {
		long info = csrel[i].info;
		csrel[i].info = (map[info >> 32] << 32) + (int)info;
	}
	for (i = 0; i < dsrel_n; i++) {
		long info = dsrel[i].info;
		dsrel[i].info = (map[info >> 32] << 32) + (int)info;
	}
	free(map);
	shdr[3].info = last + 1;
	off += 24 * syms_n;
	shdr[5].type = 1;
	shdr[5].flags = 3;
	shdr[5].offset = off;
	shdr[5].size = dsn;
	shdr[5].addralign = 0x10;
	shdr[5].entsize = 1;
	off += dsn;
	shdr[6].offset = off;
	shdr[6].type = 4;
	shdr[6].size = 24 * dsrel_n;
	shdr[6].entsize = 0x18;
	shdr[6].addralign = 8;
	shdr[6].link = 3;
	shdr[6].info = 5;
	off += 24 * dsrel_n;
	shdr[7].offset = off;
	shdr[7].type = 8;
	shdr[7].flags = 3;
	shdr[7].addralign = 0x10;
	shdr[7].entsize = 1;

	bss = 0;
	for (s = syms; s != syms + syms_n; s++) {
		if (s->shndx == 7 && s->val + s->size > bss)
			bss = s->val + s->size;
	}
	shdr[7].size = bss;
	shdr[4].type = 3;
	shdr[4].offset = off;
	shdr[4].size = symstr_n;
	shdr[4].entsize = 1;

	write(fd, &ehdr, 0x40);
	write(fd, shdr, 0x200);
	write(fd, cs, csn);
	write(fd, csrel, 24 * csrel_n);
	write(fd, syms, 24 * syms_n);
	write(fd, ds, dsn);
	write(fd, dsrel, 24 * dsrel_n);
	write(fd, symstr, symstr_n);
	free(syms);
	free(symstr);
	free(csrel);
	free(dsrel);
}
