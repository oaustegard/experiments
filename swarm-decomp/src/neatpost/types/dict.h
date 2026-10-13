#pragma once

struct iset;

struct dict {
	struct iset *set;	/* 0x00: bucket index of key hashes */
	char **keys;		/* 0x08 */
	int *vals;		/* 0x10 */
	int nkey;		/* 0x18: allocated key slots */
	int used1c;		/* 0x1c */
	int dflt;		/* 0x20: value returned when not found */
	unsigned int maxlen;		/* 0x24: prefix length hashed (tested as int, offset zero-extended) */
	int own;		/* 0x28: keys are owned (freed with dict) */
};
