# ERRORS.md — encoder-platform-survey

1. **First ladder run timed the masked-LM head, not the encoder.** Every shipped
   ONNX export except NeoBERT's ends in the vocab projection (50k for
   ModernBERT/Ettin, 256k for mmBERT). At 512 tokens the head added 38% to
   ModernBERT-base (431.5 → 311.8 ms) and tripled mmBERT-small (462.5 → 147.1
   ms). Caught by noticing the output shape was `[1, 512, vocab]`; fixed by
   cutting the graph at the final norm (`cut_heads.py`) and re-running. Direction:
   the uncut numbers would have ranked mmBERT-small below ModernBERT-base and
   understated every small model's margin over the large ones.
2. **Renaming `onnx-community` files breaks them.** `model_quantized.onnx` names
   its external data file by its original basename; the renamed copy failed to
   load until moved back into a directory under its original name. Wasted one
   run, no effect on numbers.
