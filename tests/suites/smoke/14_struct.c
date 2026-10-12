int printf(const char *fmt, ...);
struct pt { int x; int y; };
int main(void) { struct pt p = { 3, 4 }; struct pt *q = &p; q->y = 10; printf("%d %d %d\n", p.x, p.y, (int)sizeof(struct pt)); return 0; }
