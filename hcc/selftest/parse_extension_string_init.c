// EXPECT-EXIT: 0
const char s[] = __extension__ "ab" "cd";
int main(void) {
  int n = __extension__ 3;
  return (sizeof s == 5 && n == 3) ? 0 : 1;
}
