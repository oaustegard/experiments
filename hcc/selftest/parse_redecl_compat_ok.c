// EXPECT-EXIT: 5
// Compatible redeclarations (incomplete array completed, extern then definition) are fine.
int a[];
int a[3];
extern int b;
int b = 5;
int main(void) { return b; }
