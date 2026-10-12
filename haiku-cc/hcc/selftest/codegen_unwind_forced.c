// EXPECT-EXIT: 0
// EXPECT-STDOUT: cleanup 2
// EXPECT-STDOUT: cleanup 1
// _Unwind_ForcedUnwind runs the pending cleanups of the frames it passes (landing pads).
#include <unwind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static _Unwind_Reason_Code stop(int v, _Unwind_Action a, _Unwind_Exception_Class c,
                                struct _Unwind_Exception *e, struct _Unwind_Context *ctx, void *sp) {
    if (a & _UA_END_OF_STACK) abort();
    return _URC_NO_REASON;
}
static void h(void *p) { printf("cleanup %d\n", *(int *)p); fflush(stdout); }
static void last(void *p) { printf("cleanup %d\n", *(int *)p); exit(0); }
static void fu(void) {
    struct _Unwind_Exception *exc = malloc(sizeof *exc);
    memset(exc, 0, sizeof *exc);
    _Unwind_ForcedUnwind(exc, stop, 0);
    abort();
}
static void inner(void) { int a __attribute__((cleanup(h))) = 2; fu(); }
int main(void) {
    int b __attribute__((cleanup(last))) = 1;
    inner();
    return 3;
}
