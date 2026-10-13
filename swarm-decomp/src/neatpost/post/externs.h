#pragma once
/* prototypes of functions defined in other TUs of neatpost, as this TU uses them.
   Kept TU-local: other TUs define some of these with other signatures. */
int clr_get(char *s);
void dev_close(void);
int dev_mnt(int n, const char *name, const char *path);
int dev_open(const char *dir, const char *dev);
void docheader(char *title, int w, int h, int lw);
void docpagebeg(int n);
void docpageend(void);
void doctrailer(int n);
void drawbeg(void);
void drawc(int d);
void drawe(int h, int v);
void drawend(int close, int fill);
void drawa(int h, int v, int dh, int dv);
void drawl(int h, int v);
void draws(int a, int b, int c, int d);
void outc(char *s);
void outcolor(int m);
void outfont(int f);
void outgname(int n);
void outh(int v);
void outmark(int n, char *desc, int *page, int *offset, int *level);
void outmnt(int m);
void outname(int n, char *desc, int *page, int *offset);
void outpage(void);
void outrel(int h, int v);
void outset(char *name, char *val);
void outsize(int s);
void outv(int v);
