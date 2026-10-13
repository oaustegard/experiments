#pragma once
/* base type read by basetype; kind at +0, 20 bytes */
struct type {
	int kind;
	char rest[16];
};
