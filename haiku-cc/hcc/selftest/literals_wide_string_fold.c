// EXPECT-EXIT: 0
// A read of a wide literal at a constant index folds to the whole element (4 bytes
// for L/U, 2 for u); the index is a byte offset already scaled by sema.
int printf(const char *, ...);
int main(void) {
    int ok = 1;
    ok &= (1 & (L"\x1234\x5678"[1] == 0x5678));
    ok &= (3 & (u"\x1234\x5678"[1] == 0x5678));
    ok &= (5 & (U"\x1234\x5678\x9abc"[2] == 0x9abc));
    ok &= (1 & ((L"\x1234\x5678" + 1)[0] == 0x5678));
    ok &= (1 & (L"AB"[1] == 'B'));
    ok &= (1 & (u"\xffff\x0001"[0] == 0xffff));
    return ok ? 0 : 1;
}
