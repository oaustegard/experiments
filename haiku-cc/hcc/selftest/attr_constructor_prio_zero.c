// EXPECT-EXIT: 0
// EXPECT-STDOUT: b
// EXPECT-STDOUT: c
// EXPECT-STDOUT: a
// constructor(0) is an explicit priority (sorted before 101), distinct from the
// default-priority constructor, which runs last.
#include <stdio.h>
void a(void) __attribute__((constructor));
void b(void) __attribute__((constructor(0)));
void c(void) __attribute__((constructor(101)));
void a(void) { printf("a\n"); }
void b(void) { printf("b\n"); }
void c(void) { printf("c\n"); }
int main(void) { return 0; }
