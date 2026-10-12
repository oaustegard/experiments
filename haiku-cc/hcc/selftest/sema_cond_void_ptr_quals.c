// EXPECT-EXIT: 0
// ?: with a void* operand yields void* whose pointee carries the union of the
// const/volatile of both pointees (array elements looked through); _Atomic is dropped.
int main(void) {
    const int (*u)[1] = 0;
    void *v = 0;
    _Atomic void *w = 0;
    if (_Generic(1 ? u : v, const void *: 0, void *: 1) != 0) return 1;
    if (_Generic(1 ? v : u, const void *: 0, void *: 1) != 0) return 2;
    if (_Generic(1 ? w : v, _Atomic void *: 0, void *: 1) != 1) return 3;
    if (_Generic(1 ? v : w, void *: 1, default: 0) != 1) return 4;
    return 0;
}
