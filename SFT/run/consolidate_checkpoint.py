import os
import glob
import argparse
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens


def consolidate_checkpoint(ckpt_dir, output_file, use_ema=False):
    """
    Consolidates sharded FSDP checkpoints into a single model file.

    Args:
        ckpt_dir (str): Path to the FSDP checkpoint directory containing sharded files.
        output_file (str): Path to save the consolidated .safetensors file.
        use_ema (bool): If True, consolidates the 'ema' weights. Otherwise, consolidates the main 'model' weights.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    print(f"[*] Starting consolidation for checkpoint: {ckpt_dir}")

    # 1. Load configuration and instantiate a shell model on CPU
    print("[*] Loading model configuration...")
    try:
        llm_config = Qwen2Config.from_json_file(
            os.path.join(ckpt_dir, "llm_config.json")
        )
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(ckpt_dir, "vit_config.json")
        )
        # VAE config might be named differently or absent, handle this gracefully
        vae_config_path = os.path.join(ckpt_dir, "vae_config.json")
        if not os.path.exists(vae_config_path):
            vae_config_path = os.path.join(ckpt_dir, "config.json")  # Fallback name
        vae_config = (
            Qwen2Config.from_json_file(vae_config_path)
            if os.path.exists(vae_config_path)
            else None
        )

        bagel_config = BagelConfig.from_json_file(os.path.join(ckpt_dir, "config.json"))
        bagel_config.llm_config = llm_config
        bagel_config.vit_config = vit_config
        bagel_config.vae_config = vae_config

    except FileNotFoundError as e:
        print(
            f"[ERROR] Configuration file not found: {e}. Ensure all config.json files are in {ckpt_dir}"
        )
        return

    print("[*] Instantiating model on CPU...")
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config).to("cpu")

    # Resize tokenizer if new tokens were added
    tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_dir)
    tokenizer, _, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # 2. Find and consolidate sharded weights
    prefix = "ema" if use_ema else "model"
    shard_pattern = os.path.join(ckpt_dir, f"{prefix}.*.safetensors")
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        raise FileNotFoundError(
            f"No sharded files found for prefix '{prefix}' in {ckpt_dir}"
        )

    print(
        f"[*] Found {len(shard_files)} sharded files with prefix '{prefix}'. Consolidating..."
    )

    full_state_dict = {}
    for shard_file in tqdm(shard_files, desc="Consolidating shards"):
        shard_state_dict = load_file(shard_file, device="cpu")
        full_state_dict.update(shard_state_dict)

    # 3. Load consolidated weights and save the final model
    print("[*] Loading consolidated weights into the model...")
    model.load_state_dict(full_state_dict)

    print(f"[*] Saving consolidated model to {output_file}...")
    save_file(model.state_dict(), output_file)

    print(f"\n[SUCCESS] Consolidation complete. Full model saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate FSDP sharded checkpoints."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the FSDP checkpoint directory containing sharded files and configs.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the consolidated .safetensors file.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Consolidate the EMA weights instead of the main model weights.",
    )

    args = parser.parse_args()
    consolidate_checkpoint(args.ckpt_dir, args.output_file, args.use_ema)
