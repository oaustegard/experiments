// EXPECT-ERROR: [preprocess] missing terminating
// A string literal left unterminated inside #pragma is a lexical error.
#pragma message "oops
int x;
