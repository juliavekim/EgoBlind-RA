"""
setup_patches.py

One-time setup: download Kimi-VL base model from HuggingFace and patch the
modeling code in HF cache to fix compatibility issues with transformers 4.56.

Run once on Engaging after creating your conda env. Idempotent — safe to re-run.

Patches applied:
  1. is_torch_fx_available import fallback (older transformers symbol may be missing)
  2. rope_scaling fallbacks for "type" and "factor" keys
  3. tie_weights() signature accepts **kwargs
  4. inputs_embeds.clone() before assignment (gradient checkpointing fix)
  5. Remove `assert not self.training` (blocks training)
  6. .seen_tokens -> .get_seq_length() (deprecated cache API)
  7. .get_usable_length(...) -> .get_seq_length() (deprecated cache API)

Both cache locations are patched:
  - $HF_HOME/hub/.../modeling_kimi_vl.py (the file HF downloads)
  - $HF_HOME/modules/transformers_modules/.../modeling_kimi_vl.py (runtime copy)
"""

import glob
import os
import re

from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Setting up Kimi-VL on Engaging")
    print("=" * 60)

    # 1. Download model (uses HF_TOKEN from env if needed)
    print("\n[1/2] Downloading moonshotai/Kimi-VL-A3B-Instruct (~30 GB)")
    print("      Skips if already cached.")
    snapshot_download("moonshotai/Kimi-VL-A3B-Instruct")
    print("      Done.")

    # 2. Patch modeling code in BOTH cache locations
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"\n[2/2] Patching modeling code in {cache_root}")

    patterns = [
        f"{cache_root}/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/*/modeling_kimi_vl.py",
        f"{cache_root}/modules/transformers_modules/moonshotai/Kimi-VL-A3B-Instruct/*/modeling_kimi_vl.py",
    ]

    n_patched = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path) as f:
                src = f.read()

            # Patch 1: is_torch_fx_available fallback
            if "except ImportError:\n    def is_torch_fx_available" not in src:
                src = src.replace(
                    "from transformers.utils.import_utils import is_torch_fx_available",
                    "try:\n    from transformers.utils.import_utils import is_torch_fx_available\n"
                    "except ImportError:\n    def is_torch_fx_available(): return False"
                )

            # Patch 2: rope_scaling fallbacks
            src = src.replace(
                'scaling_type = self.rope_scaling["type"]',
                'scaling_type = self.rope_scaling.get("type", '
                'self.rope_scaling.get("rope_type", "default"))'
            )
            src = src.replace(
                'scaling_factor = self.rope_scaling["factor"]',
                'scaling_factor = self.rope_scaling.get("factor", 1.0)'
            )

            # Patch 3: tie_weights signature
            src = src.replace(
                "def tie_weights(self):",
                "def tie_weights(self, **kwargs):"
            )

            # Patch 4: inputs_embeds.clone() before assignment
            if ("inputs_embeds = inputs_embeds.clone()\n"
                "        inputs_embeds[input_ids == image_token_index]") not in src:
                src = src.replace(
                    'inputs_embeds[input_ids == image_token_index] = image_features',
                    'inputs_embeds = inputs_embeds.clone()\n'
                    '        inputs_embeds[input_ids == image_token_index] = image_features'
                )

            # Patch 5: Remove training-blocker assert
            src = re.sub(r'\n\s*assert not self\.training\s*\n', '\n', src)

            # Patch 6: .seen_tokens -> .get_seq_length()
            src = re.sub(r'\.seen_tokens\b', '.get_seq_length()', src)

            # Patch 7: .get_usable_length(...) -> .get_seq_length()
            src = re.sub(r'\.get_usable_length\([^)]*\)', '.get_seq_length()', src)

            with open(path, 'w') as f:
                f.write(src)

            print(f"      Patched: {path}")
            n_patched += 1

    if n_patched == 0:
        print("      WARNING: no modeling_kimi_vl.py files found to patch.")
        print(f"      Check cache_root: {cache_root}")
        print(f"      Searched patterns:")
        for p in patterns:
            print(f"        {p}")
    else:
        print(f"\n      Total files patched: {n_patched}")

    print("\n" + "=" * 60)
    print("Setup complete. You can now run generate_dpo_pairs.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()
