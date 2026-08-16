# Task: stack_vm

Write a Python module implementing a small stack-based virtual machine:

```python
class VMError(Exception): ...          # base class for all VM errors
class StackUnderflow(VMError): ...     # popping/peeking an empty operand stack
class BadJump(VMError): ...            # jump/call target out of range or into nowhere
class BadOpcode(VMError): ...          # unknown instruction name
class StepLimitExceeded(VMError): ...  # executed more than max_steps instructions
class VMZeroDivision(VMError): ...     # DIV or MOD with zero divisor

def run(program: list[tuple], max_steps: int = 10_000) -> list[int]
```

`program` is a list of instructions; each instruction is a tuple `(OP,)` or
`(OP, arg)`. `run` executes from index 0 and returns the final operand stack
(bottom first) after `HALT` or after falling off the end of the program.

The machine: one operand stack of ints, one global store (dict int->int), and a call
stack of return addresses. All arithmetic is Python int arithmetic.

Instructions (exact names, uppercase strings):

| op | arg | effect |
|---|---|---|
| `PUSH` | int | push arg |
| `POP` | – | discard top |
| `DUP` | – | duplicate top |
| `SWAP` | – | swap top two |
| `ADD` | – | pop b, pop a, push a+b |
| `SUB` | – | pop b, pop a, push a-b |
| `MUL` | – | pop b, pop a, push a*b |
| `DIV` | – | pop b, pop a, push a//b (floor); b==0 -> `VMZeroDivision` |
| `MOD` | – | pop b, pop a, push a%b (Python sign rules); b==0 -> `VMZeroDivision` |
| `NEG` | – | pop a, push -a |
| `EQ` | – | pop b, pop a, push 1 if a==b else 0 |
| `LT` | – | pop b, pop a, push 1 if a<b else 0 |
| `GT` | – | pop b, pop a, push 1 if a>b else 0 |
| `LOAD` | int | push store[arg]; missing key pushes 0 |
| `STORE` | int | pop a, store[arg] = a |
| `JMP` | int | jump to absolute index arg |
| `JZ` | int | pop a; if a==0 jump to arg, else fall through |
| `CALL` | int | push next index onto call stack, jump to arg |
| `RET` | – | pop return address from call stack, jump there; empty call stack -> `StackUnderflow` |
| `HALT` | – | stop; `run` returns the operand stack |

Precise rules:
- **Step counting**: every executed instruction (including `HALT`) counts as one
  step. If executing would exceed `max_steps` — i.e. on the (max_steps+1)-th
  instruction — raise `StepLimitExceeded` instead of executing it.
- **Jump targets** (`JMP`, `JZ` taken, `CALL`, `RET` address): valid targets are
  `0..len(program)` inclusive — jumping exactly to `len(program)` ends execution
  normally (like falling off the end). Anything negative or greater raises `BadJump`.
  An untaken `JZ` does not validate its target.
- **Operand order**: for binary ops, the top of stack is `b` (second operand):
  `PUSH 7, PUSH 3, SUB` leaves `[4]`.
- Any pop/peek on an empty operand stack raises `StackUnderflow` (this includes
  `POP`, `DUP`, `SWAP` with fewer than 2, binary ops with fewer than 2, `STORE`,
  `JZ`, `NEG`).
- An instruction whose op name is not in the table raises `BadOpcode`.
  (You may assume args, when present, are ints; missing/extra args need not be
  validated.)
- Errors are raised eagerly at the offending instruction; the partial state is
  discarded (nothing else is returned).
- Falling off the end (pc == len(program)) returns the operand stack, same as HALT.
- The store persists for the whole run; there are no frames/locals — `CALL`/`RET`
  manage only return addresses.

Examples:
- `run([("PUSH", 7), ("PUSH", 3), ("SUB",)])` -> `[4]`
- `run([("PUSH", 2), ("CALL", 4), ("HALT",), ("PUSH", 99), ("DUP",), ("MUL",), ("RET",)])`
  -> `[4]` (calls the "square" routine at index 4, returns to `HALT` at index 2)
- `run([("JMP", 0)], max_steps=10)` raises `StepLimitExceeded`

No I/O. Standard library only. The tests import `run` and all six error classes.
