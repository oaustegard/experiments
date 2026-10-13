#pragma once

/* hash table of string keys; arrays grow together via dict_extend */
struct dict {
	int *hash;	/* 0x00: bucket heads, calloc'd, nhash entries */
	int *link;	/* 0x08: chain link per slot */
	char **key;	/* 0x10: key strings */
	int *val;	/* 0x18: integer values */
	int cap;	/* 0x20: allocated slots */
	int cnt;	/* 0x24: slots used (starts at 1) */
	unsigned int nhash;	/* 0x28: number of buckets (divisor) */
	int dup;	/* 0x2c: copy keys when nonzero */
};
