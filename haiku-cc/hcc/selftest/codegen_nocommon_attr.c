// EXPECT-EXIT: 0
// __attribute__((nocommon)) on a tentative definition must go to .bss, not
// .comm, so that a weak alias of it is legal.
int foo __attribute__((nocommon));
extern int bar __attribute__((weak, alias("foo")));
int main(void) {
    return foo || bar;
}
