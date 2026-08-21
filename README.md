# tensor-copy-cost-model

A Lean 4 contract for the cost of moving tensors: **copy count does not
determine cost**. Locality does.

```bash
lake build      # no Mathlib; builds in under a second
```

## Why this exists

`seethrough-ggml`'s UNet graph emits roughly **1600 materialising copies**
(`CONT`/`CPY`) against **1524 matrix multiplies**. Read as a cost signal, that
ratio says the graph is dominated by moving tensors rather than by arithmetic,
and the obvious move is to delete copies.

That move was made. `cross_frame_block` was transposing its hidden state on the
way in and back on the way out — two full copies — to serve the one operation in
the block that needed the frame axis as the token axis. Attention already
permutes internally, so the axis choice was folded into that permutation and
both outer copies collapsed into one on attention's output. **Two copies became
one.**

The output was **bit-identical** — sha256 `f14427a0…82299` on the main PSD and
`312448031bc…` on the depth companion, matching baseline exactly. The change is
a proven semantic no-op.

It ran **6% slower**: 373.3s against 351.2s, on an RTX 4090, driver 610.88, at
res 1280 / 30 steps / seed 42.

## The mistake, made precise

The two deleted copies transposed a **contiguous** tensor: near-sequential
reads. The one added copy materialises from a worse permutation, so its
innermost contiguous run is shorter. Fewer bytes moved, more memory
transactions per byte.

`cost L n r = n / min r L` counts cache lines touched to copy `n` elements with
innermost contiguous run `r`, where a line holds `L` elements. A run shorter
than a line still pays for a whole line, which is why the divisor is `min r L`.

| theorem                           | statement                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `contiguous_is_optimal`           | no run length beats a full line                                                    |
| `strided_penalty`                 | a strided copy costs exactly `L` times its contiguous equivalent                   |
| `two_contiguous_beat_one_strided` | 2 contiguous copies < 1 strided copy, whenever `L > 2`                             |
| `copy_count_is_not_a_cost_model`  | ∃ a configuration with **fewer copies and greater cost**                           |
| `break_even`                      | replacing `c₁` contiguous copies with `c₂` strided ones wins only if `c₂ * L < c₁` |

## The rule that follows

With a 64-byte line and f32 data, `L = 16`. `break_even` then says you must
delete **at least 16 contiguous copies** to afford a single strided one. Trading
locality for copy count essentially never pays.

So an argument of the form _"this graph has too many copies, therefore removing
one makes it faster"_ is unsound on its own. It needs a locality argument: what
happens to the innermost contiguous run of every copy that remains.

## Scope, honestly

This is a **cost model, not a machine model**. It counts cache lines and
ignores prefetchers, multiple cache levels, TLB behaviour, GPU coalescing
widths, and occupancy. It is deliberately crude, because its job is to refute a
specific unsound inference — that fewer copies means less time — not to predict
runtimes. `two_contiguous_beat_one_strided` needs only `L > 2` to hold, so the
conclusion survives any plausible refinement of the constant.

The empirical numbers above are one machine, one input, one resolution. They
motivate the model; they do not validate it.

## Licence

Apache-2.0.
