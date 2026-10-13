#pragma once
/* file-scope data of the ncc TU (zero-initialised unless noted) */
static int ncc_opt = 2;
static void *arrays, *structs, *funcs, *typedefs, *locals, *globals;
static void *label_name, *label_ids;
static int arrays_n, arrays_sz, locals_n, locals_sz, globals_n, globals_sz;
static int label, label_n;
static char func_name[128];
static int nts;
static struct type ts[64];
