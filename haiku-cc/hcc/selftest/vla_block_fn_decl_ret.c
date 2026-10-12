// EXPECT-ERROR: [parse]
// Block-scope function declaration returning a pointer to a VLA is invalid.
int a;
void f(void) {
    extern int (*g(void))[a];
}
int main(void) { return 0; }
