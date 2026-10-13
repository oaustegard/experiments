#pragma once
/* mkfn back-end calls made by afm_read; kept TU-local because the mkfn TU defines mkfn_header with implicit int */
void mkfn_char(char *s, int code, int uni, int wid, int ht, int dp, int lig, int bbox);
void mkfn_header(const char *name);
void mkfn_kern(const char *left, const char *right, int v);
