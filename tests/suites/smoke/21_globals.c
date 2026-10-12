int printf(const char *fmt, ...);
int g = 5; int arr[3] = { 1, 2, 3 }; const char *msg = "glob"; static int counter;
void bump(void) { counter += g; }
int main(void) { bump(); bump(); printf("%d %d %s\n", counter, arr[2], msg); return 0; }
