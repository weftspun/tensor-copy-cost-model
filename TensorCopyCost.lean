/-
  Memory-transaction cost of strided tensor copies.

  This exists because a measured optimisation went the wrong way, and the
  reasoning that motivated it is easy to repeat.

  The setting: `seethrough-ggml`'s UNet graph emits ~1600 materialising copies
  (`CONT`/`CPY`) against ~1524 matrix multiplies. That ratio was read as "the
  graph is dominated by moving tensors", and a change was made that removed two
  copies per `cross_frame_block` and added one -- a strict reduction in copy
  *count*, and provably a semantic no-op: output was bit-identical, sha256
  f14427a0…82299 for both the baseline and the modified build.

  It ran 6% SLOWER: 373.3s against 351.2s, RTX 4090, driver 610.88, res 1280,
  30 steps, seed 42.

  The error was treating copy count as the cost model. The two deleted copies
  transposed a contiguous tensor -- near-sequential reads. The one added copy
  materialises from a worse permutation, so it reads with a shorter contiguous
  run. Fewer bytes moved; more memory transactions per byte.

  What follows makes that precise: cost is governed by the innermost contiguous
  run length, and a strict reduction in copy count can strictly increase cost.
-/

namespace TensorCopyCost

/-- Memory transactions ("cache lines touched") to copy `n` elements whose
    innermost contiguous run is `r` elements, on a machine whose cache line
    holds `L` elements.

    A run shorter than a line still pays for a whole line, so `r = 1` touches
    one line per element. Runs at least a line long amortise fully, giving a
    floor of `n / L`. Hence the divisor is `min r L`, not `r`. -/
def cost (L n r : Nat) : Nat := n / min r L

/-- Contiguous copies are optimal: no run length beats a full line. -/
theorem contiguous_is_optimal (L n r : Nat) (hr : 0 < r) (hL : 0 < L) :
    cost L n L ≤ cost L n r := by
  simp only [cost, Nat.min_self]
  exact Nat.div_le_div_left (Nat.min_le_right r L) (Nat.lt_min.mpr ⟨hr, hL⟩)

/-- A strided copy costs exactly `L` times its contiguous equivalent. This is
    the factor any copy-count saving has to beat. -/
theorem strided_penalty (L k : Nat) (hL : 0 < L) :
    cost L (L * k) 1 = L * cost L (L * k) L := by
  have h1 : min 1 L = 1 := Nat.min_eq_left hL
  simp only [cost, h1, Nat.min_self, Nat.div_one]
  rw [Nat.mul_div_cancel_left k hL]

/-- **The measured situation.**

    Two copies of a contiguous tensor cost strictly less than one copy of a
    fully-strided tensor of the same size, whenever a cache line holds more
    than two elements -- true on every real machine (a 64-byte line holds 16
    f32s).

    So deleting two contiguous copies and adding one strided copy strictly
    increases cost, while strictly decreasing copy count. -/
theorem two_contiguous_beat_one_strided
    (L k : Nat) (hL : 2 < L) (hk : 0 < k) :
    2 * cost L (L * k) L < cost L (L * k) 1 := by
  have hL0 : 0 < L := by omega
  have h1 : min 1 L = 1 := Nat.min_eq_left hL0
  simp only [cost, h1, Nat.min_self, Nat.div_one]
  rw [Nat.mul_div_cancel_left k hL0]
  -- 2 * k < L * k, from 2 < L and 0 < k, without nonlinear arithmetic:
  -- split L * k into 2 * k plus a strictly positive remainder.
  have hpos : 0 < (L - 2) * k := Nat.mul_pos (by omega) hk
  have hsplit : L * k = 2 * k + (L - 2) * k := by
    rw [← Nat.add_mul]
    have : 2 + (L - 2) = L := by omega
    rw [this]
  omega

/-- **Copy count does not determine cost.**

    There is a configuration with strictly fewer copies and strictly greater
    cost. So "this graph has too many copies, therefore removing one makes it
    faster" is unsound without a locality argument. -/
theorem copy_count_is_not_a_cost_model :
    ∃ (L n r₁ r₂ : Nat), cost L n r₂ > 2 * cost L n r₁ :=
  ⟨16, 1024, 16, 1, by decide⟩

/-- The concrete numbers: a 64-byte line holds 16 f32s, so a fully-strided copy
    costs 16x its contiguous equivalent. -/
example : cost 16 1024 1 = 16 * cost 16 1024 16 := by decide

/-- Two contiguous copies are 8x cheaper than one strided copy of the same
    size -- the exact shape of the regression this file records. -/
example : 2 * cost 16 1024 16 < cost 16 1024 1 := by decide

/-- **The break-even rule.**

    Replacing `c₁` contiguous copies with `c₂` fully-strided ones is a win only
    when `c₂ * L < c₁`. With `L = 16`, you would have to delete at least 16
    contiguous copies to afford a single strided one -- which is why trading
    locality for copy count essentially never pays. -/
theorem break_even (L k c₁ c₂ : Nat) (hL : 0 < L) :
    c₂ * cost L (L * k) 1 < c₁ * cost L (L * k) L ↔ c₂ * L * k < c₁ * k := by
  have h1 : min 1 L = 1 := Nat.min_eq_left hL
  simp only [cost, h1, Nat.min_self, Nat.div_one]
  rw [Nat.mul_div_cancel_left k hL, Nat.mul_assoc]

end TensorCopyCost
