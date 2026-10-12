// EXPECT-EXIT: 0
// Stringifying an odd run of trailing backslashes drops the last one (GCC).
#define str(x) #x
int main(void) {
  const char *a = str(\);
  const char *b = str(\\);
  const char *c = str(\\\);
  return (a[0] == 0 && b[0] == '\\' && b[1] == 0 && c[0] == '\\' && c[1] == 0) ? 0 : 1;
}
