#pragma once
#include "sbuf.h"
struct box {
	struct sbuf buf;
	int type;
	int nregd;
	int pad18;
	int cnt;
	int f20;
	int f24;
	int f28;
	int pad2c;
	int pad30;
	int pad34;
};
