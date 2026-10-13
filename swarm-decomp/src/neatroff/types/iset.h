#pragma once

/* integer set: n rows; row i holds rowlen[i] ints in tab[i] (rowcap[i] allocated) */
struct iset {
	int **tab;	/* 0x00: row pointers */
	int *rowcap;	/* 0x08: allocated size per row */
	int *rowlen;	/* 0x10: used size per row */
	int n;		/* 0x18: number of rows */
};
