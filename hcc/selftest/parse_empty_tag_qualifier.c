// EXPECT-ERROR: [parse]
// Empty declaration with a qualifier does not redeclare a visible tag of the wrong kind.
union u3 { float v; };
void h(void) { const struct u3; }
