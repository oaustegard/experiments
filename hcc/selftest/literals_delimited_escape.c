// EXPECT-EXIT: 0
// P2290 delimited escapes: \u{...}, \x{...}, \o{...}
int printf(const char *, ...);
int main(void) {
    int ok = 1;
    ok &= (U"\u{41}"[0] == 0x41);
    ok &= ("\x{41}"[0] == 'A');
    ok &= ("\o{101}"[0] == 'A');
    ok &= (L"\x{1234}"[0] == 0x1234);
    ok &= ("A"[0] == 'A');
    return ok ? 0 : 1;
}
