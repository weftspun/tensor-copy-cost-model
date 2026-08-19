import Lake
open Lake DSL

package «tensor-copy-cost-model» where
  -- Deliberately no Mathlib: the whole development is Nat arithmetic, and a
  -- gate that takes minutes to build is a gate people stop running.

lean_lib TensorCopyCost where
  roots := #[`TensorCopyCost]

@[default_target]
lean_lib Main where
  roots := #[`TensorCopyCost]
