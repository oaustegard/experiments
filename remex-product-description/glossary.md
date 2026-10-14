# Glossary

The vocabulary used across these documents. When a document uses one of these words, it means exactly this.

## The library

**Caller.** The program that imports remex and calls it. The documents are written from the caller's side; "the user" is never used.

**Call.** One invocation of a public remex function or method. A call is the unit of interaction every document narrates, in five phases: *the call is made*, *refused at once*, *the first effect*, *while it runs*, *it returns*.

## Objects

**Quantizer.** A `remex.Quantizer`. It holds a rotation and a codebook built from its encoding parameters, and is the only object that can encode, decode or search. Building one does no I/O and reads no data.

**Container.** A `CompressedVectors` or `PackedVectors`: the codes for *n* vectors plus whatever per-row floats the mode needs (`norms`, and `mean_dots` when centred), and the recorded encoding parameters (`d`, `bits`, `rotation`, `mean`). A container cannot decode itself; it needs a compatible quantizer.

**Codes.** The per-coordinate integers a quantizer writes, `bits` wide. In memory a `CompressedVectors` holds them one per byte; on disk and in a `PackedVectors` they are bit-packed.

**Container file.** A file `save` wrote: a NumPy `.npz` archive holding the packed codes, the per-row floats, and the recorded parameters. Arrow and `.pq` files are other formats with their own documents.

## Encoding parameters

**Encoding parameters.** The values that determine which codes a vector gets: `d`, `bits`, `seed`, `rotation`, `normalize`, `scale`, and `mean` when given. Two quantizers with the same encoding parameters write identical codes for the same input and read each other's codes correctly. `renorm` is not an encoding parameter: it changes how codes are read, not what they are.

**Recorded parameter.** An encoding parameter that a container carries with it: `d`, `bits`, `rotation`, the mode (recorded as whether `norms` is present) and `mean`. A quantizer compares four of them against its own before decoding or searching: `bits`, `rotation`, the mode and `mean`. `d` is recorded but not compared. `seed` and `scale` are encoding parameters that are *not* recorded in a container or a container file.

**Guarded.** A recorded parameter the quantizer compares, refusing the call on a mismatch: `bits`, `rotation`, the mode and `mean`.

**Rotation.** Which orthogonal transform the quantizer applies before quantizing: `"rht"` (randomized Hadamard; the default for even `d`), `"haar"` (the default for odd `d`, and the value a file with no rotation record is read as), or `"none"`.

**Seed.** The integer that selects one rotation out of the family. Default 42.

## Modes

**Normalizing mode.** The default (`normalize=True`). Each vector is split into a unit direction, which is quantized, and a float32 norm, which is stored exactly.

**Scalar mode.** `normalize=False`. Coordinates are quantized directly against a codebook cut for the caller-declared `scale`; no norms are stored. A container's mode is recorded by whether it has `norms`.

**Centred.** A quantizer built with `mean=`, and the containers it writes. The codes describe each vector's offset from the declared corpus mean, and the container carries the mean and a `mean_dots` column. remex never measures the mean itself.

## Precision

**Precision.** The bit level a decode or search reads the codes at, from 1 to the quantizer's `bits`. Lower precisions read the top bits of each code (Matryoshka nesting). `precision=None` means full precision.

## Events that end a call

**Returns.** The call completed and its effects are all in place.

**Refused.** The call raised before changing anything the caller can observe: no file written or truncated, no cache built, no object mutated. Every documented mismatch check refuses.

**Fails partway.** The call raised, or the process died, after the first effect. What state is left behind (a truncated file, a half-built cache) is what each document's interrupt table records.

**Silently wrong.** The call returns normally with a result that does not mean what the caller assumes: vectors decoded against the wrong rotation family, for instance. These documents treat every silently wrong case as a suspected bug unless the code documents it as intended.

## Files

**Overwrite.** Saving to a path that already holds a file. remex never asks; the old file is truncated when the new write begins, not replaced when it ends.
