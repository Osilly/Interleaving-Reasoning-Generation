# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import base64
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import (
    ParquetStandardIterableDataset,
    InterleavedBaseIterableDataset,
)
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class MultiThinkInterleaveT2IDataset(
    InterleavedBaseIterableDataset, ParquetStandardIterableDataset
):
    """
    A dataset that handles interleaved text and image data in a conversational format.
    Each row in the Parquet file is expected to have a 'conversation' column,
    which contains a JSON string representing a list of turns.
    The `need_loss` field in each part of the conversation explicitly controls
    whether loss is calculated for that part.

    Example of a 'conversation' JSON structure:
    [
        {
            "role": "user",
            "content": "Generate a picture of a cat on a skateboard.",
            "need_loss": false
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "content": "Okay, first I'll sketch the cat.",
                    "need_loss": true
                },
                {
                    "type": "image",
                    "content": "<binary_image_data_of_cat_sketch>",
                    "need_loss": false
                },
                {
                    "type": "text",
                    "content": "Now, I'll add the skateboard and final details.",
                    "need_loss": true
                },
                {
                    "type": "image",
                    "content": "<binary_image_data_of_final_image>",
                    "need_loss": true
                }
            ]
        }
    ]
    For an image part, `need_loss: false` implies it's a condition for subsequent
    steps (requiring VAE and ViT processing), while `need_loss: true` means it's a
    generation target.
    """

    def parse_row(self, row):
        if "conversation" not in row:
            return {}

        data = self._init_data()

        try:
            conversation = json.loads(row["conversation"])
        except (json.JSONDecodeError, TypeError):
            # Skip rows with invalid JSON
            return {}

        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")

            if not role or not content:
                continue

            if role == "user":
                if isinstance(content, str):
                    need_loss = turn.get("need_loss", False)
                    data = self._add_text(data, content, need_loss=need_loss)

            elif role == "assistant":
                if not isinstance(content, list):
                    continue

                for part in content:
                    part_type = part.get("type")
                    part_content = part.get("content")

                    if not part_type or not part_content:
                        continue

                    if part_type == "text":
                        need_loss = part.get("need_loss", True)
                        data = self._add_text(data, part_content, need_loss=need_loss)

                    elif part_type == "image":
                        try:
                            # Decode the base64 string to get the image bytes
                            decoded_content = base64.b64decode(part_content)
                            image = pil_img2rgb(Image.open(io.BytesIO(decoded_content)))

                            # Directly read flags from the data, with safe defaults
                            need_loss = part.get("need_loss", False)
                            need_vae = part.get("need_vae", False)
                            need_vit = part.get("need_vit", False)

                            # Add a check for vit_transform if need_vit is true
                            if need_vit and self.vit_transform is None:
                                raise ValueError(
                                    "Dataset requires ViT processing, but 'vit_transform' was not provided."
                                )

                            data = self._add_image(
                                data,
                                image,
                                need_loss=need_loss,
                                need_vae=need_vae,
                                need_vit=need_vit,
                            )
                        except Exception as e:
                            # Skip parts with invalid image data
                            # print(f"Error parsing image part: {e}") # Optional: for debugging
                            continue

        return data
