// EXPECT-ERROR: [parse]
struct g { int w; int x[]; } g;
char t[sizeof (g.x) ? -1 : 1];
int main(void) { return 0; }
