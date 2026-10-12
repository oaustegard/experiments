// EXPECT-ERROR: empty character constant
// A wide or u8 character constant with no characters is rejected (C11 6.4.4.4).
int main(void) { int a = u''; return a; }
