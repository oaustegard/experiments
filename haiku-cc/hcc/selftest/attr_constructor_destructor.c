// EXPECT-EXIT: 0
// EXPECT-STDOUT: prio-101
// EXPECT-STDOUT: default-prio
// EXPECT-STDOUT: main
// EXPECT-STDOUT: fini
#include <stdio.h>
static void early(void) __attribute__((constructor));
static void early(void) { puts("default-prio"); }
static void ctor(void) __attribute__((constructor(101)));
static void ctor(void) { puts("prio-101"); }
static void dtor(void) __attribute__((destructor));
static void dtor(void) { puts("fini"); }
int main(void) { puts("main"); return 0; }
