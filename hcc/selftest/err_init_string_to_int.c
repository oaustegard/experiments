// EXPECT-ERROR: [parse]
// A string literal (an array) cannot initialize an integer scalar element.
char g[] = { [7] = "abcd" };
