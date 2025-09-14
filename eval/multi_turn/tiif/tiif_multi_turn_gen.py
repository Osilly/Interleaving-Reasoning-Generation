# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from pathlib import Path
import logging

import torch
import torch.multiprocessing as mp
from SFT.data.data_utils import add_special_tokens
from SFT.modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from SFT.modeling.qwen2 import Qwen2Tokenizer
from SFT.modeling.autoencoder import load_ae

import copy
from PIL import Image
from SFT.modeling.bagel.qwen2_navit import NaiveCache
from tqdm import tqdm
import random
import numpy as np
from queue import Empty


def setup_worker(rank, world_size):
    """Sets up the environment for each worker process."""
    total_gpus = torch.cuda.device_count()
    print(f"[Worker-{rank}] Total GPUs available: {total_gpus}")

    if rank >= total_gpus:
        print(f"[Worker-{rank}] WARNING: rank {rank} >= total GPUs {total_gpus}")
        # Use modulo to wrap around available GPUs
        gpu_id = rank % total_gpus
    else:
        gpu_id = rank

    # Set environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)  # Set to the specific GPU
        print(f"[Worker-{rank}] Set CUDA device to: {torch.cuda.current_device()}")
        print(f"[Worker-{rank}] Device name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"[Worker-{rank}] Physical GPU ID: {gpu_id}")
    else:
        print(f"[Worker-{rank}] CUDA not available!")

    return gpu_id


def cleanup():
    """Cleans up the process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_tiif_data(input_dir):
    """Loads TIIF-Bench data from jsonl files."""
    data = []
    jsonl_files = []

    # Find all jsonl files in the input directory
    for file_path in Path(input_dir).glob("*.jsonl"):
        jsonl_files.append(file_path)

    print(f"Found {len(jsonl_files)} jsonl files")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    item = json.loads(line)
                    # Add metadata for tracking
                    item["file_name"] = jsonl_file.name
                    item["line_idx"] = idx
                    data.append(item)

    print(f"Loaded {len(data)} items from TIIF-Bench")
    return data


def writer_fn(queue, output_dir, total_workers, model_name="BAGEL-7B-MoT-MultiTurn"):
    """A dedicated writer process that pulls results from a queue and saves them."""
    finished_workers = 0
    results = []

    while finished_workers < total_workers:
        try:
            # Set a timeout to prevent waiting indefinitely
            record = queue.get(timeout=3600)
            if record == "DONE":
                finished_workers += 1
            else:
                results.append(record)
                # Save images according to TIIF-Bench directory structure
                data_type = record["type"]
                file_name = record["file_name"]
                line_idx = record["line_idx"]

                # Create TIIF-Bench compatible directory structure
                # output/<dimension>/<model_name>/short_description/turn1_<idx>.png
                # output/<dimension>/<model_name>/short_description/turn2_<idx>.png
                # output/<dimension>/<model_name>/long_description/turn1_<idx>.png
                # output/<dimension>/<model_name>/long_description/turn2_<idx>.png

                type_dir = os.path.join(output_dir, data_type, model_name)
                short_desc_dir = os.path.join(type_dir, "short_description")
                long_desc_dir = os.path.join(type_dir, "long_description")

                os.makedirs(short_desc_dir, exist_ok=True)
                os.makedirs(long_desc_dir, exist_ok=True)

                # Save short description results
                if record.get("short_results"):
                    short_results = record["short_results"]
                    if short_results.get("turn_1", {}).get("image"):
                        turn1_path = os.path.join(
                            short_desc_dir, f"turn1_{line_idx}.png"
                        )
                        short_results["turn_1"]["image"].save(turn1_path)

                        # Save thinking text
                        turn1_txt_path = os.path.join(
                            short_desc_dir, f"turn1_{line_idx}.txt"
                        )
                        with open(turn1_txt_path, "w", encoding="utf-8") as f:
                            f.write(short_results["turn_1"].get("text", ""))

                    if short_results.get("turn_2", {}).get("image"):
                        turn2_path = os.path.join(
                            short_desc_dir, f"turn2_{line_idx}.png"
                        )
                        short_results["turn_2"]["image"].save(turn2_path)

                        # Save thinking text
                        turn2_txt_path = os.path.join(
                            short_desc_dir, f"turn2_{line_idx}.txt"
                        )
                        with open(turn2_txt_path, "w", encoding="utf-8") as f:
                            f.write(short_results["turn_2"].get("text", ""))

                # Save long description results
                if record.get("long_results"):
                    long_results = record["long_results"]
                    if long_results.get("turn_1", {}).get("image"):
                        turn1_path = os.path.join(
                            long_desc_dir, f"turn1_{line_idx}.png"
                        )
                        long_results["turn_1"]["image"].save(turn1_path)

                        # Save thinking text
                        turn1_txt_path = os.path.join(
                            long_desc_dir, f"turn1_{line_idx}.txt"
                        )
                        with open(turn1_txt_path, "w", encoding="utf-8") as f:
                            f.write(long_results["turn_1"].get("text", ""))

                    if long_results.get("turn_2", {}).get("image"):
                        turn2_path = os.path.join(
                            long_desc_dir, f"turn2_{line_idx}.png"
                        )
                        long_results["turn_2"]["image"].save(turn2_path)

                        # Save thinking text
                        turn2_txt_path = os.path.join(
                            long_desc_dir, f"turn2_{line_idx}.txt"
                        )
                        with open(turn2_txt_path, "w", encoding="utf-8") as f:
                            f.write(long_results["turn_2"].get("text", ""))

        except Empty:
            print(
                f"[Writer] Timed out after 1 hour of inactivity. Assuming workers have died."
            )
            break  # Exit if queue is empty for a long time
        except Exception as e:
            print(
                f"[Writer] An unexpected error occurred: {e}. Remaining workers: {total_workers - finished_workers}"
            )
            break

    print("[Writer] Writer process finished.")


GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here.
After generating an image and completing one round of thinking, you should continue to analyze and think further, then generate another improved image. For each subsequent round, carefully compare the previously generated image with the new one, identify specific visual differences and improvements, and generate a detailed "think process" that explains how to transform the previous image into the improved version.
The think process for each improvement round must follow this structure:
1. Start with <think> and end with </think>
2. Have clear paragraph breaks with the following sections:
   - Initial analysis paragraph (acknowledge shortcomings of the previous image)
   - ### Detailed Explanation of Required Improvements:
   - ### Step-by-Step Modification Guidance:
   - ### Final Comprehensive Prompt for the Improved Image:
Each section should provide concrete, actionable guidance for text-to-image generation, focusing on composition, lighting, colors, textures, positioning, and style. Do not mention the improved image directly; only provide the think process. Repeat this process for each new image, ensuring each think process is specific enough to guide precise image improvements.
"""


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device with consistent dtype
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            # Ensure consistent dtype - use bfloat16 for floating point tensors
            if v.is_floating_point():
                generation_input[k] = v.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input[k] = v.to(device=device)
    return generation_input


def generate_multi_turn_image_with_think(
    prompt,
    num_timesteps=50,
    # Turn 1 parameters
    cfg_text_scale_turn1=4.0,
    cfg_interval_turn1=[0.4, 1.0],
    cfg_renorm_min_turn1=0.0,
    cfg_renorm_type_turn1="global",
    timestep_shift_turn1=3.0,
    # Turn 2 parameters
    cfg_text_scale_turn2=4.0,
    cfg_img_scale_turn2=2.0,
    cfg_interval_turn2=[0, 1.0],
    cfg_renorm_min_turn2=0.0,
    cfg_type_turn2="serial_text_img",
    cfg_renorm_type_turn2="text_channel",
    timestep_shift_turn2=3.0,
    # Common parameters
    resolution=1024,
    max_length=1000,
    device=None,
    model=None,
    vae_model=None,
    tokenizer=None,
    new_token_ids=None,
):
    """
    Generate two images in a multi-turn fashion.
    1. think -> image
    2. think -> image (based on the first image)
    """
    h, w = resolution, resolution
    output_results = {
        "turn_1": {"text": None, "image": None},
        "turn_2": {"text": None, "image": None},
    }

    with torch.no_grad():
        # =================================================================
        # Initialization
        # =================================================================
        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # Add system prompt
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[GEN_THINK_SYSTEM_PROMPT],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # Prepare CFG context
        cfg_generation_input = model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
        )
        cfg_generation_input = move_generation_input_to_device(
            cfg_generation_input, device
        )

        # =================================================================
        # Turn 1: Generate first thought and image from prompt
        # =================================================================
        print("--- Turn 1: Generating first thought and image ---")

        # Update context with the user prompt
        cfg_text_past_key_values = copy.deepcopy(past_key_values)
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # Generate first thinking text
        tmp_past_key_values = copy.deepcopy(past_key_values)
        tmp_newlens = copy.deepcopy(newlens)
        tmp_new_rope = copy.deepcopy(new_rope)

        tmp_generation_input = model.prepare_start_tokens(
            tmp_newlens, tmp_new_rope, new_token_ids
        )
        tmp_generation_input = move_generation_input_to_device(
            tmp_generation_input, device
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = model.generate_text(
                past_key_values=tmp_past_key_values,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=new_token_ids["eos_token_id"],
                **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:, 0])
        think1 = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        output_results["turn_1"]["text"] = think1
        print(f"Thinking 1: {think1}")

        # Update context with the first thinking text
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[think1],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # Generate the first image
        generation_input = model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale_turn1,
                cfg_interval=cfg_interval_turn1,
                timestep_shift=timestep_shift_turn1,
                cfg_renorm_min=cfg_renorm_min_turn1,
                cfg_renorm_type=cfg_renorm_type_turn1,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_position_ids=cfg_generation_input[
                    "cfg_packed_position_ids"
                ],
                cfg_text_key_values_lens=cfg_generation_input["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=cfg_generation_input[
                    "cfg_packed_query_indexes"
                ],
                cfg_text_packed_key_value_indexes=cfg_generation_input[
                    "cfg_packed_key_value_indexes"
                ],
                **generation_input,
            )

        latent1 = unpacked_latent[0]
        latent1 = latent1.reshape(1, h // 16, w // 16, 2, 2, 16)
        latent1 = torch.einsum("nhwpqc->nchpwq", latent1)
        latent1 = latent1.reshape(1, 16, h // 8, w // 8)
        # Ensure latent is in the correct dtype before VAE decoding
        latent1 = latent1.to(device=device, dtype=torch.bfloat16)
        image1 = vae_model.decode(latent1)
        tmpimage1 = (
            ((image1 * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        tmpimage1 = Image.fromarray(tmpimage1)
        output_results["turn_1"]["image"] = tmpimage1
        print("Image 1 generated.")

        # =================================================================
        # Turn 2: Generate second thought and image based on the first
        # =================================================================
        print("\n--- Turn 2: Generating second thought and image ---")

        # Update context with the first image as a condition
        # Convert image to the format expected by the model
        from SFT.data.transforms import ImageTransform

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        image1_for_context = tmpimage1.copy()

        # Add the first image to the context for both VAE and ViT
        # This is the key step that was missing!

        # Update context with image for VAE (generation)
        generation_input_vae, newlens, new_rope = model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image1_for_context],
            transforms=vae_transform,
            new_token_ids=new_token_ids,
        )
        generation_input_vae = move_generation_input_to_device(
            generation_input_vae, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_vae(
                vae_model, past_key_values, **generation_input_vae
            )

        # Update context with image for ViT (understanding)
        generation_input_vit, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image1_for_context],
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        generation_input_vit = move_generation_input_to_device(
            generation_input_vit, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_vit(
                past_key_values, **generation_input_vit
            )

        # Prepare contexts for the second turn
        # NOTE: For the second turn, we need to create a new CFG context that includes the image
        # but excludes the upcoming thinking text, so we use the current state
        cfg_text_past_key_values = copy.deepcopy(past_key_values)

        # Update CFG generation input for the second turn to match the new sequence length
        cfg_generation_input_turn2 = model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
        )
        cfg_generation_input_turn2 = move_generation_input_to_device(
            cfg_generation_input_turn2, device
        )

        # Prepare cfg_img context (following demo.py pattern)
        # NOTE: cfg_img_past_key_values should NOT contain images - it's for "no image" CFG
        cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
        cfg_img_newlens = [0]
        cfg_img_new_rope = [0]

        # Add system prompt to cfg_img context
        cfg_img_generation_input, cfg_img_newlens, cfg_img_new_rope = (
            model.prepare_prompts(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                prompts=[GEN_THINK_SYSTEM_PROMPT],
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
        )
        cfg_img_generation_input = move_generation_input_to_device(
            cfg_img_generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_img_past_key_values = model.forward_cache_update_text(
                cfg_img_past_key_values, **cfg_img_generation_input
            )

        # Add the original user prompt to cfg_img context (no image context)
        cfg_img_generation_input, cfg_img_newlens, cfg_img_new_rope = (
            model.prepare_prompts(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                prompts=[prompt],
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
        )
        cfg_img_generation_input = move_generation_input_to_device(
            cfg_img_generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_img_past_key_values = model.forward_cache_update_text(
                cfg_img_past_key_values, **cfg_img_generation_input
            )

        # Add the first thinking text to cfg_img context
        cfg_img_generation_input, cfg_img_newlens, cfg_img_new_rope = (
            model.prepare_prompts(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                prompts=[think1],
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
        )
        cfg_img_generation_input = move_generation_input_to_device(
            cfg_img_generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_img_past_key_values = model.forward_cache_update_text(
                cfg_img_past_key_values, **cfg_img_generation_input
            )

        # Generate second thinking text
        tmp_past_key_values = copy.deepcopy(past_key_values)
        tmp_newlens = copy.deepcopy(newlens)
        tmp_new_rope = copy.deepcopy(new_rope)

        tmp_generation_input = model.prepare_start_tokens(
            tmp_newlens, tmp_new_rope, new_token_ids
        )
        tmp_generation_input = move_generation_input_to_device(
            tmp_generation_input, device
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = model.generate_text(
                past_key_values=tmp_past_key_values,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=new_token_ids["eos_token_id"],
                **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:, 0])
        think2 = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        output_results["turn_2"]["text"] = think2
        print(f"Thinking 2: {think2}")

        # Update context with the second thinking text
        generation_input, newlens, new_rope = model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[think2],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # Also add the second thinking text to cfg_img context
        cfg_img_generation_input, cfg_img_newlens, cfg_img_new_rope = (
            model.prepare_prompts(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                prompts=[think2],
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
        )
        cfg_img_generation_input = move_generation_input_to_device(
            cfg_img_generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_img_past_key_values = model.forward_cache_update_text(
                cfg_img_past_key_values, **cfg_img_generation_input
            )

        # Prepare cfg_img generation input
        cfg_img_generation_input_turn2 = model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            image_sizes=[(h, w)],
        )
        cfg_img_generation_input_turn2 = move_generation_input_to_device(
            cfg_img_generation_input_turn2, device
        )

        # Generate the second image
        generation_input = model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
            new_token_ids=new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale_turn2,
                cfg_img_scale=cfg_img_scale_turn2,
                cfg_interval=cfg_interval_turn2,
                timestep_shift=timestep_shift_turn2,
                cfg_renorm_min=cfg_renorm_min_turn2,
                cfg_type=cfg_type_turn2,
                cfg_renorm_type=cfg_renorm_type_turn2,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_text_packed_position_ids=cfg_generation_input_turn2[
                    "cfg_packed_position_ids"
                ],
                cfg_text_key_values_lens=cfg_generation_input_turn2[
                    "cfg_key_values_lens"
                ],
                cfg_text_packed_query_indexes=cfg_generation_input_turn2[
                    "cfg_packed_query_indexes"
                ],
                cfg_text_packed_key_value_indexes=cfg_generation_input_turn2[
                    "cfg_packed_key_value_indexes"
                ],
                cfg_img_packed_position_ids=cfg_img_generation_input_turn2[
                    "cfg_packed_position_ids"
                ],
                cfg_img_key_values_lens=cfg_img_generation_input_turn2[
                    "cfg_key_values_lens"
                ],
                cfg_img_packed_query_indexes=cfg_img_generation_input_turn2[
                    "cfg_packed_query_indexes"
                ],
                cfg_img_packed_key_value_indexes=cfg_img_generation_input_turn2[
                    "cfg_packed_key_value_indexes"
                ],
                **generation_input,
            )

        latent2 = unpacked_latent[0]
        latent2 = latent2.reshape(1, h // 16, w // 16, 2, 2, 16)
        latent2 = torch.einsum("nhwpqc->nchpwq", latent2)
        latent2 = latent2.reshape(1, 16, h // 8, w // 8)
        # Ensure latent is in the correct dtype before VAE decoding
        latent2 = latent2.to(device=device, dtype=torch.bfloat16)
        image2 = vae_model.decode(latent2)
        tmpimage2 = (
            ((image2 * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        tmpimage2 = Image.fromarray(tmpimage2)
        output_results["turn_2"]["image"] = tmpimage2
        print("Image 2 generated.")

    return output_results


def worker_fn(rank, world_size, args, data_chunks, queue):
    """
    The worker function for each process.
    Loads a model on a specific GPU and processes a chunk of data.
    """
    setup_worker(rank, world_size)

    # Each worker selects its own data chunk based on its rank
    data_chunk = data_chunks[rank]

    # Suppress the verbose logging from the diffusers library
    logging.getLogger("diffusers").setLevel(logging.WARNING)

    # Set seed for reproducibility
    seed = 42 + rank  # Different seed for each worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Use the actual GPU rank instead of cuda:0
    device = f"cuda:{rank}"

    print(f"[GPU-{rank}] Loading model from {args.model_path}...")
    try:
        # Load model configurations and initialize model
        llm_config = Qwen2Config.from_json_file(
            os.path.join(args.model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(args.model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(
            local_path=os.path.join(args.model_path, "ae.safetensors")
        )

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=args.max_latent_size,
        )

        # Initialize model with empty weights to avoid memory issues
        from accelerate import init_empty_weights

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Use accelerate to load model with proper device mapping and dtype
        print(f"[GPU-{rank}] Loading model weights with accelerate...")
        from accelerate import load_checkpoint_and_dispatch

        # Create simple device map for single GPU
        device_map = {"": device}

        # Load checkpoint and dispatch with consistent dtype
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(args.model_path, "ema_bf16.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload",
        )

        # Move VAE model to device and set dtype
        vae_model = vae_model.to(device=device, dtype=torch.bfloat16)

        # Set to eval mode
        model = model.eval()
        vae_model = vae_model.eval()

        print(f"[GPU-{rank}] Model loaded and converted to bfloat16")

    except Exception as e:
        print(f"[GPU-{rank}] Error initializing model: {e}")
        queue.put("DONE")  # Signal completion even on error
        cleanup()
        return

    print(f"[GPU-{rank}] Model loaded. Starting inference on {len(data_chunk)} items.")

    # Setup progress bar for this worker
    progress_bar = tqdm(total=len(data_chunk), position=rank, desc=f"GPU-{rank}")

    num_timesteps = 50

    for item in data_chunk:
        data_type = item.get("type")
        short_description = item.get("short_description")
        long_description = item.get("long_description")
        file_name = item.get("file_name")
        line_idx = item.get("line_idx")

        if not short_description or not long_description or data_type is None:
            progress_bar.update(1)
            continue

        print(f"[GPU-{rank}] Processing {data_type}: '{short_description[:50]}...'")

        try:
            # Generate multi-turn images for short description
            short_results = generate_multi_turn_image_with_think(
                prompt=short_description,
                num_timesteps=num_timesteps,
                # Turn 1 parameters
                cfg_text_scale_turn1=args.cfg_text_scale_turn1,
                cfg_interval_turn1=args.cfg_interval_turn1,
                cfg_renorm_min_turn1=args.cfg_renorm_min_turn1,
                cfg_renorm_type_turn1=args.cfg_renorm_type_turn1,
                timestep_shift_turn1=args.timestep_shift_turn1,
                # Turn 2 parameters
                cfg_text_scale_turn2=args.cfg_text_scale_turn2,
                cfg_img_scale_turn2=args.cfg_img_scale_turn2,
                cfg_interval_turn2=args.cfg_interval_turn2,
                cfg_renorm_min_turn2=args.cfg_renorm_min_turn2,
                cfg_type_turn2=args.cfg_type_turn2,
                cfg_renorm_type_turn2=args.cfg_renorm_type_turn2,
                timestep_shift_turn2=args.timestep_shift_turn2,
                # Common parameters
                resolution=args.resolution,
                max_length=4096,
                device=device,
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )

            # Generate multi-turn images for long description
            long_results = generate_multi_turn_image_with_think(
                prompt=long_description,
                num_timesteps=num_timesteps,
                # Turn 1 parameters
                cfg_text_scale_turn1=args.cfg_text_scale_turn1,
                cfg_interval_turn1=args.cfg_interval_turn1,
                cfg_renorm_min_turn1=args.cfg_renorm_min_turn1,
                cfg_renorm_type_turn1=args.cfg_renorm_type_turn1,
                timestep_shift_turn1=args.timestep_shift_turn1,
                # Turn 2 parameters
                cfg_text_scale_turn2=args.cfg_text_scale_turn2,
                cfg_img_scale_turn2=args.cfg_img_scale_turn2,
                cfg_interval_turn2=args.cfg_interval_turn2,
                cfg_renorm_min_turn2=args.cfg_renorm_min_turn2,
                cfg_type_turn2=args.cfg_type_turn2,
                cfg_renorm_type_turn2=args.cfg_renorm_type_turn2,
                timestep_shift_turn2=args.timestep_shift_turn2,
                # Common parameters
                resolution=args.resolution,
                max_length=4096,
                device=device,
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )

            if short_results and long_results:
                # Create result record
                result_record = {
                    "type": data_type,
                    "file_name": file_name,
                    "line_idx": line_idx,
                    "short_description": short_description,
                    "long_description": long_description,
                    "short_results": short_results,
                    "long_results": long_results,
                }

                # Put result into the queue
                queue.put(result_record)
            else:
                print(
                    f"[GPU-{rank}] Failed to generate images for {data_type}: {short_description[:50]}..."
                )

        except Exception as e:
            print(
                f"[GPU-{rank}] Error during inference for {data_type}: {short_description[:50]}...: {e}"
            )

        progress_bar.update(1)

    progress_bar.close()
    queue.put("DONE")  # Signal that this worker is finished
    print(f"[GPU-{rank}] Worker finished.")
    cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn images for TIIF-Bench using Bagel model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing TIIF-Bench jsonl files.",
    )
    # Turn 1 parameters
    parser.add_argument(
        "--cfg_text_scale_turn1",
        type=float,
        default=4.0,
        help="CFG text scale for turn 1",
    )
    parser.add_argument(
        "--cfg_interval_turn1",
        type=float,
        nargs=2,
        default=[0.4, 1.0],
        help="CFG interval for turn 1",
    )
    parser.add_argument(
        "--cfg_renorm_min_turn1",
        type=float,
        default=0.0,
        help="CFG renorm min for turn 1",
    )
    parser.add_argument(
        "--cfg_renorm_type_turn1",
        type=str,
        default="global",
        help="CFG renorm type for turn 1",
    )
    parser.add_argument(
        "--timestep_shift_turn1",
        type=float,
        default=3.0,
        help="Timestep shift for turn 1",
    )

    # Turn 2 parameters
    parser.add_argument(
        "--cfg_text_scale_turn2",
        type=float,
        default=4.0,
        help="CFG text scale for turn 2",
    )
    parser.add_argument(
        "--cfg_img_scale_turn2",
        type=float,
        default=2.0,
        help="CFG image scale for turn 2",
    )
    parser.add_argument(
        "--cfg_interval_turn2",
        type=float,
        nargs=2,
        default=[0, 1.0],
        help="CFG interval for turn 2",
    )
    parser.add_argument(
        "--cfg_renorm_min_turn2",
        type=float,
        default=0.0,
        help="CFG renorm min for turn 2",
    )
    parser.add_argument(
        "--cfg_type_turn2",
        type=str,
        default="serial_text_img",
        help="CFG type for turn 2",
    )
    parser.add_argument(
        "--cfg_renorm_type_turn2",
        type=str,
        default="text_channel",
        help="CFG renorm type for turn 2",
    )
    parser.add_argument(
        "--timestep_shift_turn2",
        type=float,
        default=3.0,
        help="Timestep shift for turn 2",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--model_path", type=str, default="hf/BAGEL-7B-MoT/")
    parser.add_argument("--model_name", type=str, default="BAGEL-7B-MoT-MultiTurn")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and split data
    print("Loading TIIF-Bench data...")
    data = load_tiif_data(args.input_dir)
    if not data:
        print("No valid data found in the input directory.")
        return

    # Check GPU availability
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if available_gpus == 0:
        print("❌ No CUDA GPUs available!")
        return

    world_size = args.num_gpus
    if world_size > available_gpus:
        print(
            f"Warning: --num_gpus ({world_size}) is greater than available GPUs ({available_gpus}). Using all available GPUs."
        )
        world_size = available_gpus

    data_chunks = [data[i::world_size] for i in range(world_size)]
    print(
        f"Data loaded. Total items: {len(data)}. Distributing among {world_size} GPUs."
    )

    # Print data distribution
    for i, chunk in enumerate(data_chunks):
        print(f"GPU-{i}: {len(chunk)} items")

    # Use a manager queue for inter-process communication
    manager = mp.Manager()
    queue = manager.Queue()

    # Start the writer process
    writer_process = mp.Process(
        target=writer_fn, args=(queue, args.output_dir, world_size, args.model_name)
    )
    writer_process.start()

    # Spawn worker processes
    mp.spawn(
        worker_fn,  # type: ignore
        args=(world_size, args, data_chunks, queue),
        nprocs=world_size,
        join=True,
    )

    # Wait for the writer to finish
    writer_process.join()

    print(
        f"TIIF-Bench multi-turn inference completed! Results saved to {args.output_dir}"
    )


if __name__ == "__main__":
    # Ensure we use 'spawn' start method for CUDA safety
    mp.set_start_method("spawn", force=True)
    main()
