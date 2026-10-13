#pragma once
/* type entry; 20 bytes. kind at +0; the rest is either raw bytes (main.c
   clears them) or the fields used by the type stack ts[] (flags, ptr, arr, tag) */
struct type {
	int kind;	/* +0 */
	union {
		char rest[16];	/* +4 */
		struct {
			int flags;	/* +4: bits 0 and 2 are tested */
			int ptr;	/* +8: nonzero marks a pointer/array operand */
			int arr;	/* +12: index into the arrays table */
			int tag;	/* +16 */
		};
	};
};
