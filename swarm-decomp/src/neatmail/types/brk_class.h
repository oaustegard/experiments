#pragma once

/* one entry of brk_classes[]: 16 bytes, from the brk_classes data listing */
struct brk_class {
	const char *name;	/* 0x00: ":alpha:" */
	const char *chars;	/* 0x08: "a-zA-Z" */
};
