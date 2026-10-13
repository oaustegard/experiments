#pragma once
/* mkfn back-end calls made by afm_read; kept TU-local because the mkfn TU defines mkfn_header with implicit int */
void mkfn_char(char *name, int idx, int code, int wid, int x0, int y0, int x1, int y1);
void mkfn_header(const char *name);
void mkfn_kern(const char *left, const char *right, int v);
