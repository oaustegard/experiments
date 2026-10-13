#pragma once
/* the cdef table: number of user-defined strings and their replacement pointers (tr/cdef_add.c).
   The TU alone never writes these, so gcc would fold the reads to their zero initial values;
   taking their addresses into an escaping table keeps the reads (the table is never used). */
static int cdef_n;
static char *cdef_dst[128];
static int cdef_fn[128];
static char cdef_src[128][32];
static int cdef_expanding;
void *cdef_esc[5] = { &cdef_n, cdef_dst, cdef_fn, cdef_src, &cdef_expanding };
