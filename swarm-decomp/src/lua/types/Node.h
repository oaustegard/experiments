#pragma once
#include "lobject.h"

struct NodeKey {
  Value value_;
  lu_byte tt_;
  lu_byte key_tt;
  int next;
  Value key_val;
};

typedef union Node {
  struct NodeKey u;
  TValue i_val;
} Node;
