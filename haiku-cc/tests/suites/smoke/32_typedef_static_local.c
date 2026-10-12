int printf(const char *fmt, ...);
typedef struct node { int v; struct node *next; } Node;
int counter(void) { static int n = 0; return ++n; }
int main(void) { Node c = { 3, 0 }, b = { 2, &c }, a = { 1, &b }; int s = 0; for (Node *p = &a; p; p = p->next) s += p->v; counter(); counter(); printf("%d %d\n", s, counter()); return 0; }
