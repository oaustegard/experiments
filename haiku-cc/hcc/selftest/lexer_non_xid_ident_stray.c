// EXPECT-ERROR: stray
// EXPECT-FLAGS: -std=c23
// In C23 a non-XID character (superscript two) cannot start or continue an identifier
// (gcc -std=c2x -pedantic-errors: "stray '\\302' in program"). Before C23 gcc accepts it
// as a C11 Annex D character (lex_c11_ident_start.c).
int ² = 42;
