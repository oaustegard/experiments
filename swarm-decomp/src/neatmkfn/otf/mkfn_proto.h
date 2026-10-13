#pragma once
/* mkfn helpers this TU calls; kept TU-local (mkfn's own definitions are not shared types) */
int mkfn_script(const char *tag, int n);
int mkfn_lang(const char *tag, int n);
int mkfn_featrank(const char *name, const char *tag);
int mkfn_font(const char *name);
void mkfn_char(char *name, int idx, int code, int wid, int x0, int y0, int x1, int y1);
void mkfn_header(const char *name);
void mkfn_kern(const char *left, const char *right, int v);
extern int mkfn_warn;
