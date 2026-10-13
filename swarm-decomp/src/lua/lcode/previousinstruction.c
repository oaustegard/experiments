#include "FuncState.h"
#include "Instruction.h"

static Instruction *previousinstruction (FuncState *fs) {
  static const Instruction invalidinstruction = ~(Instruction)0;
  if (fs->pc > fs->lasttarget)
    return &fs->f->code[fs->pc - 1];
  else
    return (Instruction *)&invalidinstruction;
}
