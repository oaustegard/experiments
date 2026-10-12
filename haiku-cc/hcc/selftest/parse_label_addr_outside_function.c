// EXPECT-ERROR: [parse]
// GNU label address (&&z) in a file-scope declarator is outside any function.
int k (int [sizeof &&z]);
