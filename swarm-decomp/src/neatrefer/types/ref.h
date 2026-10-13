#pragma once
/* one bibliographic reference record (sizeof 0x808).
 * f[c]   field value for the field letter c (f['J'] = journal, f['B'] = booktitle, ...)
 * au[i]  author name i, i < nau
 * seq    index assigned when the record is cited (-1 when not yet)  */
struct ref {
	char *f[128];
	char *au[128];
	int seq;
	int nau;
};
