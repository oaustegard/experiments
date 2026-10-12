// EXPECT-ERROR: [parse]
// copy() with a string literal argument is rejected, also in [[gnu::copy(...)]] form.
int [[gnu::copy ("")]] a;
