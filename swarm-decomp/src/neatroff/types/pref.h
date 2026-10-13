#pragma once

/* prefix table: an iset plus parallel key/value arrays */
struct pref {
	void *set;	/* 0x00: iset handle */
	char **key;	/* 0x08 */
	int *val;	/* 0x10 */
	int cap;	/* 0x18 */
	int cnt;	/* 0x1c */
};
