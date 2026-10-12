int printf(const char *fmt, ...);
int f(int x) { switch (x) { case 1: return 10; case 2: case 3: return 20; default: return 30; } }
int main(void) { printf("%d %d %d %d\n", f(1), f(2), f(3), f(9)); return 0; }
