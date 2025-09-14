# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint,
    FSDPConfig,
    grad_checkpoint_check_fn,
    fsdp_wrapper,
    fsdp_ema_setup,
    fsdp_ema_update,
)

import shutil


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={
            "help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."
        },
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."},
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."},
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={
            "help": "Path to the pretrained VAE checkpoint for latent-space image generation."
        },
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={
            "help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."
        },
    )
    max_latent_size: int = field(
        default=32,
        metadata={
            "help": "Maximum latent grid size (patches per side) for the VAE latent tensor."
        },
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."},
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."},
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={
            "help": "Maximum number of ViT patches along one image side after cropping / resize."
        },
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={
            "help": "Activation function used in the latent-to-text connector MLP."
        },
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={
            "help": "Interpolate positional embeddings when image resolution differs from pre-training."
        },
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={
            "help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."
        },
    )
    vit_rope: bool = field(
        default=False, metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."},
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."},
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={
            "help": "Probability of dropping ViT visual features during training."
        },
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={
            "help": "YAML file specifying dataset groups, weights, and preprocessing rules."
        },
    )
    prefetch_factor: int = field(
        default=2,
        metadata={
            "help": "How many batches each DataLoader worker pre-loads in advance."
        },
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."},
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={
            "help": "Maximum tokens allowed in one raw sample; longer samples are skipped."
        },
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={
            "help": "While batch length is below this, pop from the overflow buffer before new sampling."
        },
    )
    max_buffer_size: int = field(
        default=50,
        metadata={
            "help": "Maximum number of oversized samples kept in the overflow buffer."
        },
    )
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used when shuffling / sampling data shards to ensure reproducibility."
        },
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True, metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True, metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results", metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."},
    )
    wandb_project: str = field(
        default="bagel", metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."},
    )
    wandb_runid: str = field(
        default="0",
        metadata={
            "help": "Unique identifier to resume a previous W&B run, if desired."
        },
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."},
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."},
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."},
    )
    auto_resume: bool = field(
        default=False,
        metadata={
            "help": "Automatically pick up the latest checkpoint found in checkpoint_dir."
        },
    )
    resume_from: str = field(
        default=None,
        metadata={
            "help": "Explicit checkpoint path to resume from (overrides auto_resume)."
        },
    )
    resume_model_only: bool = field(
        default=False,
        metadata={
            "help": "Load only model weights, ignoring optimizer/scheduler states."
        },
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={
            "help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."
        },
    )
    finetune_from_hf: bool = field(
        default=False, metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10, metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000, metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."},
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."},
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."},
    )
    lr: float = field(
        default=1e-4, metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={
            "help": "Minimum learning rate for cosine schedule (ignored for constant)."
        },
    )
    beta1: float = field(default=0.9, metadata={"help": "AdamW β₁ coefficient."})
    beta2: float = field(default=0.95, metadata={"help": "AdamW β₂ coefficient."})
    eps: float = field(
        default=1e-15, metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={
            "help": "Decay rate for the exponential moving average of model weights."
        },
    )
    max_grad_norm: int = field(
        default=1.0, metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={
            "help": "Shift applied to diffusion timestep indices (for latent prediction)."
        },
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."},
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."},
    )
    text_loss_disable_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to disable text loss calculation. Set to -1 to never disable."
        },
    )
    image_loss_disable_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to disable image loss calculation. Set to -1 to never disable."
        },
    )
    freeze_llm_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to freeze language model parameters. Set to -1 to never freeze."
        },
    )
    freeze_text_experts_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to freeze only text/understanding expert parameters in MoE. Set to -1 to never freeze."
        },
    )
    freeze_vit_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to freeze ViT and connector parameters. Set to -1 to never freeze."
        },
    )
    freeze_image_experts_step: int = field(
        default=-1,
        metadata={
            "help": "Step after which to freeze image generation expert parameters in MoE. Set to -1 to never freeze."
        },
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={
            "help": "Reweight CE loss by token importance (provided via ce_loss_weights)."
        },
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={
            "help": "Soft target token count; yield the batch once it reaches or exceeds this size."
        },
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={
            "help": "Number of model replicas per GPU rank for tensor parallelism."
        },
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."},
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={
            "help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."
        },
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={
            "help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."
        },
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."},
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."
        },
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."},
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={
            "help": "Duplicate initial MoE experts so each has identical initialisation."
        },
    )
    use_flex: bool = field(
        default=False,
        metadata={
            "help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."
        },
    )


def copy_pretrained_files(pretrained_path: str, checkpoint_dir: str, logger):
    """
    Copy all files from pretrained path to checkpoint directory, excluding training-generated files

    Args:
        pretrained_path: Path to pretrained model
        checkpoint_dir: Checkpoint save directory
        logger: Logger instance
    """
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained path does not exist: {pretrained_path}")
        return

    # Training-generated files that should not be overwritten by pretrained files
    training_files = {
        "ema.safetensors",
        "model.safetensors",
        "scheduler.pt",
        "data_status.pt",
    }

    # Add optimizer file patterns
    import glob

    optimizer_files = glob.glob(os.path.join(checkpoint_dir, "optimizer.*.pt"))
    training_files.update([os.path.basename(f) for f in optimizer_files])

    copied_count = 0
    skipped_count = 0

    for item in os.listdir(pretrained_path):
        source_path = os.path.join(pretrained_path, item)
        target_path = os.path.join(checkpoint_dir, item)

        # Skip training-generated files
        if item in training_files:
            logger.info(f"Skipping training-generated file: {item}")
            skipped_count += 1
            continue

        # Skip directories (modify here if recursive copy is needed)
        if os.path.isdir(source_path):
            logger.info(f"Skipping directory: {item}")
            continue

        try:
            # Copy file
            shutil.copy2(source_path, target_path)
            logger.info(f"Copying file: {item}")
            copied_count += 1
        except Exception as e:
            logger.warning(f"Failed to copy file {item}: {e}")

    logger.info(
        f"File copy completed: copied {copied_count} files, skipped {skipped_count} files"
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_args.model_path, "llm_config.json")
        )
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(
            model_args.llm_path, config=llm_config
        )
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(model_args.model_path, "vit_config.json")
            )
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = (
            vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        )
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(
                model_args.vit_path, config=vit_config
            )

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=(
                os.path.join(model_args.model_path, "ae.safetensors")
                if training_args.finetune_from_hf
                else model_args.vae_path
            )
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, vit_model if training_args.visual_und else None, config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_path if training_args.finetune_from_hf else model_args.llm_path
    )
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn,
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = (
            FSDPCheckpoint.try_load_train_state(
                resume_from,
                optimizer,
                scheduler,
                fsdp_config,
            )
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # train loop
    start_time = time()
    logger.info(
        f"Training for {training_args.total_steps} steps, starting at {train_step}..."
    )
    for curr_step, data in enumerate(train_loader, start=train_step):
        # Check if language model should be frozen
        if (
            training_args.freeze_llm_step >= 0
            and curr_step >= training_args.freeze_llm_step
            and not training_args.freeze_llm
        ):
            logger.info(f"Freezing language model parameters at step {curr_step}")

            # For FSDP, freeze language_model flat_params
            frozen_llm_flat_params = 0
            for name, param in fsdp_model.named_parameters():
                if "_flat_param" in name and "language_model" in name:
                    param.requires_grad = False
                    frozen_llm_flat_params += 1
                    logger.info(f"Frozen LLM flat param: {name}")

            logger.info(f"Frozen {frozen_llm_flat_params} LLM flat parameters")

            # Also try module-level freezing as backup
            fsdp_model.language_model.eval()
            module_llm_freeze_count = 0
            for param in fsdp_model.language_model.parameters():
                param.requires_grad = False
                module_llm_freeze_count += 1

            logger.info(f"Module-level LLM frozen {module_llm_freeze_count} parameters")
            training_args.freeze_llm = True  # Mark as frozen to avoid repeated freezing

        # Check if only text experts should be frozen (more precise control)
        if (
            training_args.freeze_text_experts_step >= 0
            and curr_step >= training_args.freeze_text_experts_step
            and not getattr(training_args, "text_experts_frozen", False)
        ):
            logger.info(
                f"Freezing text/understanding expert parameters at step {curr_step}"
            )
            # For FSDP, we need a different approach since parameters are flattened
            # We'll use a more targeted approach by identifying specific flat_params to freeze

            frozen_flat_params = 0
            for name, param in fsdp_model.named_parameters():
                if "_flat_param" in name:
                    should_freeze = False

                    # Freeze language model components that are text/understanding-specific
                    if "language_model" in name:
                        # A parameter is a text expert if it's in the language model
                        # but NOT part of a generation-specific module.
                        # All generation-specific modules contain 'moe_gen' in their name.
                        if "moe_gen" not in name:
                            should_freeze = True

                    if should_freeze:
                        param.requires_grad = False
                        frozen_flat_params += 1
                        logger.info(f"Frozen text expert flat param: {name}")

            logger.info(f"Frozen {frozen_flat_params} text expert flat parameters")

            # Alternative approach: try to freeze individual modules and see if it affects flat_params
            logger.info("Also trying module-level freezing...")
            module_freeze_count = 0

            # Freeze text-related modules
            if hasattr(fsdp_model, "language_model"):
                # Freeze embedding
                if hasattr(fsdp_model.language_model, "model"):
                    if hasattr(fsdp_model.language_model.model, "embed_tokens"):
                        for param in fsdp_model.language_model.model.embed_tokens.parameters():
                            param.requires_grad = False
                            module_freeze_count += 1
                
                # Freeze lm_head
                if hasattr(fsdp_model.language_model, "lm_head"):
                    for param in fsdp_model.language_model.lm_head.parameters():
                        param.requires_grad = False
                        module_freeze_count += 1
                
                # Freeze model layers (text-related parts)
                if hasattr(fsdp_model.language_model, "model"):
                    if hasattr(fsdp_model.language_model.model, "layers"):
                        for layer in fsdp_model.language_model.model.layers:
                            # Only freeze text-related parts, exclude generation components
                            for name, param in layer.named_parameters():
                                if (
                                    "moe_gen" not in name
                                    and "time_embedder" not in name
                                    and "vae2llm" not in name
                                    and "llm2vae" not in name
                                    and "latent_pos_embed" not in name
                                ):
                                    param.requires_grad = False
                                    module_freeze_count += 1
                
                # Freeze norm layers
                if hasattr(fsdp_model.language_model, "model"):
                    if hasattr(fsdp_model.language_model.model, "norm"):
                        for param in fsdp_model.language_model.model.norm.parameters():
                            param.requires_grad = False
                            module_freeze_count += 1

            logger.info(f"Module-level text experts frozen {module_freeze_count} parameters")

            training_args.text_experts_frozen = True  # Mark as frozen

            # # Verify freezing by counting frozen parameters
            # frozen_params = 0
            # total_params = 0
            # frozen_param_names = []
            # for name, param in fsdp_model.named_parameters():
            #     total_params += 1
            #     if not param.requires_grad:
            #         frozen_params += 1
            #         frozen_param_names.append(name)

            # logger.info(
            #     f"Frozen {frozen_params}/{total_params} parameters ({frozen_params/total_params*100:.1f}%)"
            # )
            # logger.info(f"Frozen parameter names: {frozen_param_names}")

            # # Also check a few specific parameters to see if they were frozen
            # test_params = [
            #     "language_model.model.embed_tokens.weight",
            #     "language_model.lm_head.weight",
            #     "language_model.model.norm.weight",
            # ]
            # for param_name in test_params:
            #     try:
            #         param = dict(fsdp_model.named_parameters())[param_name]
            #         logger.info(f"{param_name}: requires_grad={param.requires_grad}")
            #     except KeyError:
            #         logger.info(f"{param_name}: not found in named_parameters()")

            # # Count parameters in each component
            # component_counts = {}
            # for name, param in fsdp_model.named_parameters():
            #     component = name.split(".")[0] if "." in name else name
            #     if component not in component_counts:
            #         component_counts[component] = {"total": 0, "frozen": 0}
            #     component_counts[component]["total"] += 1
            #     if not param.requires_grad:
            #         component_counts[component]["frozen"] += 1

            # for component, counts in component_counts.items():
            #     logger.info(f"{component}: {counts['frozen']}/{counts['total']} frozen")

        # Check if ViT and connector should be frozen
        if (
            training_args.freeze_vit_step >= 0
            and curr_step >= training_args.freeze_vit_step
            and not getattr(training_args, "vit_frozen", False)
        ):
            logger.info(f"Freezing ViT and connector parameters at step {curr_step}")

            # For FSDP, freeze ViT-related flat_params
            frozen_vit_flat_params = 0
            for name, param in fsdp_model.named_parameters():
                if "_flat_param" in name:
                    should_freeze = False

                    # Freeze ViT-related components
                    if (
                        "vit_model" in name
                        or "connector" in name
                        or "vit_pos_embed" in name
                    ):
                        should_freeze = True

                    if should_freeze:
                        param.requires_grad = False
                        frozen_vit_flat_params += 1
                        logger.info(f"Frozen ViT flat param: {name}")

            logger.info(f"Frozen {frozen_vit_flat_params} ViT flat parameters")

            # Also try module-level freezing as backup
            module_vit_freeze_count = 0
            if hasattr(fsdp_model, "vit_model") and fsdp_model.vit_model is not None:
                # Access through FSDP wrapper if needed
                vit_model = fsdp_model.vit_model
                if hasattr(vit_model, "_fsdp_wrapped_module"):
                    vit_model = vit_model._fsdp_wrapped_module
                vit_model.eval()
                for param in fsdp_model.vit_model.parameters():
                    param.requires_grad = False
                    module_vit_freeze_count += 1

            # Freeze connector
            if hasattr(fsdp_model, "connector"):
                for param in fsdp_model.connector.parameters():
                    param.requires_grad = False
                    module_vit_freeze_count += 1

            # Freeze ViT position embedding
            if hasattr(fsdp_model, "vit_pos_embed"):
                for param in fsdp_model.vit_pos_embed.parameters():
                    param.requires_grad = False
                    module_vit_freeze_count += 1

            logger.info(f"Module-level ViT frozen {module_vit_freeze_count} parameters")

            training_args.vit_frozen = True  # Mark as frozen

        # Check if image generation experts should be frozen
        if (
            training_args.freeze_image_experts_step >= 0
            and curr_step >= training_args.freeze_image_experts_step
            and not getattr(training_args, "image_experts_frozen", False)
        ):
            logger.info(f"Freezing image generation expert parameters at step {curr_step}")

            # For FSDP, freeze image generation-related flat_params
            frozen_image_experts_flat_params = 0
            for name, param in fsdp_model.named_parameters():
                if "_flat_param" in name:
                    should_freeze = False

                    # Freeze image generation-related components
                    # Check for root-level generation components
                    if (
                        "time_embedder" in name
                        or "vae2llm" in name
                        or "llm2vae" in name
                        or "latent_pos_embed" in name
                    ):
                        should_freeze = True
                    
                    # Check for language_model moe_gen components
                    if "language_model" in name and "moe_gen" in name:
                        should_freeze = True

                    if should_freeze:
                        param.requires_grad = False
                        frozen_image_experts_flat_params += 1
                        logger.info(f"Frozen image expert flat param: {name}")

            logger.info(f"Frozen {frozen_image_experts_flat_params} image expert flat parameters")

            # Also try module-level freezing as backup
            module_image_experts_freeze_count = 0

            # Freeze generation-specific modules if they exist
            # Note: These components are at the root level of fsdp_model, not under language_model
            
            # Try to freeze time_embedder (at root level)
            if hasattr(fsdp_model, "time_embedder"):
                for param in fsdp_model.time_embedder.parameters():
                    param.requires_grad = False
                    module_image_experts_freeze_count += 1

            # Try to freeze vae2llm connector (at root level)
            if hasattr(fsdp_model, "vae2llm"):
                for param in fsdp_model.vae2llm.parameters():
                    param.requires_grad = False
                    module_image_experts_freeze_count += 1

            # Try to freeze llm2vae connector (at root level)
            if hasattr(fsdp_model, "llm2vae"):
                for param in fsdp_model.llm2vae.parameters():
                    param.requires_grad = False
                    module_image_experts_freeze_count += 1

            # Try to freeze latent_pos_embed (at root level)
            if hasattr(fsdp_model, "latent_pos_embed"):
                for param in fsdp_model.latent_pos_embed.parameters():
                    param.requires_grad = False
                    module_image_experts_freeze_count += 1

            # Try to freeze moe_gen components within language_model layers
            if hasattr(fsdp_model, "language_model"):
                if hasattr(fsdp_model.language_model, "model"):
                    if hasattr(fsdp_model.language_model.model, "layers"):
                        for layer in fsdp_model.language_model.model.layers:
                            # Access the actual layer through FSDP wrapper
                            actual_layer = layer
                            if hasattr(layer, "_fsdp_wrapped_module"):
                                actual_layer = layer._fsdp_wrapped_module
                            if hasattr(actual_layer, "_checkpoint_wrapped_module"):
                                actual_layer = actual_layer._checkpoint_wrapped_module
                            
                            # Freeze moe_gen components in each layer
                            for name, param in actual_layer.named_parameters():
                                if "moe_gen" in name:
                                    param.requires_grad = False
                                    module_image_experts_freeze_count += 1
                    
                    # Freeze norm_moe_gen at model level
                    if hasattr(fsdp_model.language_model.model, "norm_moe_gen"):
                        for param in fsdp_model.language_model.model.norm_moe_gen.parameters():
                            param.requires_grad = False
                            module_image_experts_freeze_count += 1

            logger.info(f"Module-level image experts frozen {module_image_experts_freeze_count} parameters")

            training_args.image_experts_frozen = True  # Mark as frozen

        # # Lightweight parameter statistics without summon_full_params
        # if dist.get_rank() == 0:
        #     frozen_params = 0
        #     total_params = 0
        #     module_stats = {}
        #     frozen_param_names = []

        #     for name, param in fsdp_model.named_parameters():
        #         total_params += 1

        #         # Extract module name (first part before the first dot)
        #         module_name = name.split(".")[0] if "." in name else name
        #         if module_name not in module_stats:
        #             module_stats[module_name] = {"total": 0, "frozen": 0}
        #         module_stats[module_name]["total"] += 1

        #         if not param.requires_grad:
        #             frozen_params += 1
        #             module_stats[module_name]["frozen"] += 1
        #             frozen_param_names.append(name)

        #     # Overall statistics
        #     logger.info(
        #         f"Overall: Frozen {frozen_params}/{total_params} parameters ({frozen_params/total_params*100:.1f}%)"
        #     )

        #     # Per-module statistics
        #     logger.info("Per-module freezing statistics:")
        #     for module_name, stats in sorted(module_stats.items()):
        #         frozen_ratio = (
        #             stats["frozen"] / stats["total"] * 100 if stats["total"] > 0 else 0
        #         )
        #         logger.info(
        #             f"  {module_name}: {stats['frozen']}/{stats['total']} ({frozen_ratio:.1f}%) frozen"
        #         )

        #     # List of all frozen parameters
        #     if frozen_param_names:
        #         logger.info(f"All frozen parameters ({len(frozen_param_names)} total):")
        #         for name in frozen_param_names:
        #             logger.info(f"  - {name}")

        data = data.cuda(device).to_dict()
        data_indexes = data.pop("batch_data_indexes", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)
        # Store original ce data for monitoring if needed
        text_loss_disabled = (
            training_args.text_loss_disable_step >= 0
            and curr_step >= training_args.text_loss_disable_step
        )
        image_loss_disabled = (
            training_args.image_loss_disable_step >= 0
            and curr_step >= training_args.image_loss_disable_step
        )
        original_ce_loss_indexes = data.get("ce_loss_indexes", None)
        original_packed_label_ids = data.get("packed_label_ids", None)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data["padded_latent"] = vae_model.encode(data.pop("padded_images"))

            # Check if text loss should be disabled for training
            if text_loss_disabled:
                # Disable text loss calculation for training
                data["ce_loss_indexes"] = None
                data["packed_label_ids"] = None

            # Check if text loss should be disabled for training
            if image_loss_disabled:
                # Disable text loss calculation for training
                data["mse_loss_indexes"] = None
                # data["packed_timesteps"] = None

            loss_dict = fsdp_model(**data)

            # If text loss is disabled, compute it separately for monitoring
            if text_loss_disabled and original_ce_loss_indexes is not None:
                with torch.no_grad():
                    # Restore original data for monitoring calculation
                    data["ce_loss_indexes"] = original_ce_loss_indexes
                    data["packed_label_ids"] = original_packed_label_ids
                    monitor_loss_dict = fsdp_model(**data)
                    loss_dict["ce"] = monitor_loss_dict["ce"]

        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            # Use original ce_loss_indexes for token counting if text loss is disabled
            ce_loss_indexes_for_count = (
                original_ce_loss_indexes
                if text_loss_disabled
                else data["ce_loss_indexes"]
            )
            total_ce_tokens = torch.tensor(
                len(ce_loss_indexes_for_count), device=device
            )
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()

            # Only add to training loss if text loss is not disabled
            if not text_loss_disabled:
                loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            if not image_loss_disabled:
                mse = loss_dict["mse"]
                total_mse_tokens = torch.tensor(
                    len(data["mse_loss_indexes"]), device=device
                )
                dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
                loss_dict["mse"] = mse.detach()
                loss = loss + mse * training_args.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0, device=device)
                total_mse_tokens = torch.tensor(0, device=device)
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data["sample_lens"]), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["total_mse_tokens"] = total_mse_tokens.item()
            wandb_log["total_ce_tokens"] = total_ce_tokens.item()
            wandb_log["total_norm"] = total_norm.item()
            wandb_log["total_samples"] = total_samples.item()

            mem_allocated = torch.tensor(
                torch.cuda.max_memory_allocated() / 1024**2, device=device
            )
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log["mem_allocated"] = mem_allocated
            mem_cache = torch.tensor(
                torch.cuda.max_memory_reserved() / 1024**2, device=device
            )
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log["mem_cache"] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item["dataset_name"] not in data_status.keys():
                data_status[item["dataset_name"]] = {}
            data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
            )

            if dist.get_rank() == 0:
                ckpt_dir = os.path.join(
                    training_args.checkpoint_dir, f"{curr_step:07d}"
                )

                # Copy all config files from pretrain path
                if training_args.finetune_from_hf:
                    copy_pretrained_files(model_args.model_path, ckpt_dir, logger)
                else:
                    # Copy config files from LLM path for non-HF finetuning
                    copy_pretrained_files(model_args.llm_path, ckpt_dir, logger)
            
            dist.barrier()

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
