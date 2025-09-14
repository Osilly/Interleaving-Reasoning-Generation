# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import json
import glob
import openai
import time
import re
from tqdm import tqdm
import math
import random
import concurrent.futures
import threading

raw_prompt = """
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_1 = """
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
The order of answers must correspond exactly to the order of the questions.
Each question must have only one answer. Output one answer if there is only one question.
Directly return the answers to each question, without any additional content.
Each answer must be on its own line!
Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_2 = """
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
"""

# Multi-turn specific prompts for comparing images
multi_turn_comparison_prompt = """
You are tasked with examining two images generated in a multi-turn process. The first image was generated from an initial prompt, and the second image was generated after the model analyzed and improved upon the first image.

Please answer the following questions about BOTH images:

For Image 1 (First Turn):
##YNQuestions##

For Image 2 (Second Turn):
##YNQuestions##

Additionally, please answer these comparison questions:
- Is the second image an improvement over the first image in terms of visual quality?
- Is the second image more accurate to the prompt than the first image?
- Does the second image show better composition than the first image?

Instructions:
1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. First answer all questions for Image 1, then all questions for Image 2, then the comparison questions.
3. Maintain the exact order of the questions in your answers.
4. Provide only one answer per question.
5. Return only the answers—no additional commentary.
6. Each answer must be on its own line.
"""


def load_jsonl_lines(jsonl_file):
    lines = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj)
            except Exception as e:
                print(f"[Warning] Parse line error in {jsonl_file}: {e}")
    return lines


def generate_with_prompt(prompt, image_path, client, model="gpt-4o", image_path_2=None):
    import base64

    # Prepare the content list
    content = [{"type": "text", "text": prompt}]

    # Add first image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        }
    )

    # Add second image if provided (for multi-turn comparison)
    if image_path_2:
        with open(image_path_2, "rb") as image_file:
            image_data_2 = base64.b64encode(image_file.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_2}"},
            }
        )

    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {"role": "user", "content": content},
    ]

    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=1.0
    )

    return completion.choices[0].message.content


def format_questions_prompt(raw_prompt, questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    prompt_template = random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])
    formatted_prompt = prompt_template.replace("##YNQuestions##", formatted_questions)
    return formatted_prompt


def format_multi_turn_questions_prompt(questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    formatted_prompt = multi_turn_comparison_prompt.replace(
        "##YNQuestions##", formatted_questions
    )
    return formatted_prompt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_multi_turn_images_by_idx(img_dir, idx):
    """Find both turn1 and turn2 images for a given index"""
    turn1_pattern = os.path.join(img_dir, f"turn1_{idx}.*")
    turn2_pattern = os.path.join(img_dir, f"turn2_{idx}.*")

    turn1_files = [
        f
        for f in glob.glob(turn1_pattern)
        if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))
    ]
    turn2_files = [
        f
        for f in glob.glob(turn2_pattern)
        if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))
    ]

    if not turn1_files:
        raise FileNotFoundError(f"No turn1 image found for index {idx} in {img_dir}")
    if not turn2_files:
        raise FileNotFoundError(f"No turn2 image found for index {idx} in {img_dir}")

    return turn1_files[0], turn2_files[0]


def find_image_by_idx(img_dir, idx):
    """Find single image by index (for backward compatibility)"""
    pattern = os.path.join(img_dir, f"{idx}.*")
    files = [
        f
        for f in glob.glob(pattern)
        if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))
    ]
    if files:
        return files[0]  # 返回路径字符串
    else:
        raise FileNotFoundError(f"No image found for index {idx} in {img_dir}")


def collect_multi_turn_tasks(
    jsonl_dir, image_dir, eval_model, output_dir, sample_idx_file=None, postfix=""
):
    """Collect tasks for multi-turn evaluation"""
    tasks = []
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    if sample_idx_file is not None:
        with open(sample_idx_file, "r") as f:
            sample_idx = json.load(f)

    for jsonl_file in jsonl_files:
        attribute = os.path.splitext(os.path.basename(jsonl_file))[0]
        lines = load_jsonl_lines(jsonl_file)
        attr_type = lines[0]["type"]
        if sample_idx_file is not None:
            line_indices = sample_idx[attr_type]
            lines = [lines[idx] for idx in line_indices]
        else:
            line_indices = list(range(len(lines)))

        for desc in ["long_description", "short_description"]:
            img_dir = os.path.join(image_dir, attr_type + postfix, eval_model, desc)
            out_dir = os.path.join(
                output_dir,
                eval_model,
                attr_type,
                "long" if desc.startswith("long") else "short",
            )
            ensure_dir(out_dir)

            for idx, line in zip(line_indices, lines):
                try:
                    # For multi-turn, we need both turn1 and turn2 images
                    turn1_img_path, turn2_img_path = find_multi_turn_images_by_idx(
                        img_dir, idx
                    )

                    # Create separate output files for individual turn evaluation and comparison
                    out_path_turn1 = os.path.join(out_dir, f"turn1_{idx}.json")
                    out_path_turn2 = os.path.join(out_dir, f"turn2_{idx}.json")
                    out_path_comparison = os.path.join(
                        out_dir, f"comparison_{idx}.json"
                    )

                    # Skip if all files exist
                    if (
                        os.path.exists(out_path_turn1)
                        and os.path.exists(out_path_turn2)
                        and os.path.exists(out_path_comparison)
                    ):
                        continue

                except Exception as e:
                    print(e)
                    continue

                tasks.append(
                    {
                        "attribute": attr_type,
                        "desc": desc,
                        "jsonl_file": jsonl_file,
                        "line_idx": idx,
                        "jsonl_line": line,
                        "turn1_img_path": turn1_img_path,
                        "turn2_img_path": turn2_img_path,
                        "out_path_turn1": out_path_turn1,
                        "out_path_turn2": out_path_turn2,
                        "out_path_comparison": out_path_comparison,
                    }
                )
    return tasks


class OutputFormatError(Exception):
    pass


def extract_yes_no(model_output, questions):
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(yes|no)\b", line.strip(), flags=re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue
    if len(preds) != len(questions):
        raise OutputFormatError(
            f"Preds count {len(preds)} != questions count {len(questions)}"
        )
    return preds


def extract_multi_turn_yes_no(model_output, questions):
    """Extract yes/no answers for multi-turn comparison (2x questions + 3 comparison questions)"""
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(yes|no)\b", line.strip(), flags=re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue

    expected_count = (
        len(questions) * 2 + 3
    )  # questions for turn1 + questions for turn2 + 3 comparison questions
    if len(preds) != expected_count:
        raise OutputFormatError(
            f"Preds count {len(preds)} != expected count {expected_count}"
        )

    # Split predictions
    turn1_preds = preds[: len(questions)]
    turn2_preds = preds[len(questions) : len(questions) * 2]
    comparison_preds = preds[len(questions) * 2 :]

    return turn1_preds, turn2_preds, comparison_preds


def process_single_task(task, client, model, eval_turn, lock=None):
    """Process a single evaluation task with error handling"""
    try:
        item = task["jsonl_line"]
        questions = item.get("yn_question_list", [])
        gt_answers = item.get("yn_answer_list", [])

        # Evaluate based on eval_turn parameter
        if eval_turn in ["1", "both"] and not os.path.exists(task["out_path_turn1"]):
            # Evaluate turn 1
            prompt = format_questions_prompt(raw_prompt, questions)
            model_output = generate_with_prompt(
                prompt, task["turn1_img_path"], client, model=model
            )
            model_pred = extract_yes_no(model_output, questions)
            result = {
                "attribute": task["attribute"],
                "desc": task["desc"],
                "turn": 1,
                "jsonl_file": os.path.basename(task["jsonl_file"]),
                "line_idx": task["line_idx"],
                "questions": questions,
                "gt_answers": gt_answers,
                "model_pred": model_pred,
                "model_output": model_output,
            }

            # Use lock if provided (for concurrent execution)
            if lock:
                with lock:
                    with open(task["out_path_turn1"], "w", encoding="utf-8") as fout:
                        json.dump(result, fout, ensure_ascii=False, indent=2)
            else:
                with open(task["out_path_turn1"], "w", encoding="utf-8") as fout:
                    json.dump(result, fout, ensure_ascii=False, indent=2)

        if eval_turn in ["2", "both"] and not os.path.exists(task["out_path_turn2"]):
            # Evaluate turn 2
            prompt = format_questions_prompt(raw_prompt, questions)
            model_output = generate_with_prompt(
                prompt, task["turn2_img_path"], client, model=model
            )
            model_pred = extract_yes_no(model_output, questions)
            result = {
                "attribute": task["attribute"],
                "desc": task["desc"],
                "turn": 2,
                "jsonl_file": os.path.basename(task["jsonl_file"]),
                "line_idx": task["line_idx"],
                "questions": questions,
                "gt_answers": gt_answers,
                "model_pred": model_pred,
                "model_output": model_output,
            }

            # Use lock if provided (for concurrent execution)
            if lock:
                with lock:
                    with open(task["out_path_turn2"], "w", encoding="utf-8") as fout:
                        json.dump(result, fout, ensure_ascii=False, indent=2)
            else:
                with open(task["out_path_turn2"], "w", encoding="utf-8") as fout:
                    json.dump(result, fout, ensure_ascii=False, indent=2)

        if eval_turn == "both" and not os.path.exists(task["out_path_comparison"]):
            # Evaluate comparison between turns
            prompt = format_multi_turn_questions_prompt(questions)
            model_output = generate_with_prompt(
                prompt,
                task["turn1_img_path"],
                client,
                model=model,
                image_path_2=task["turn2_img_path"],
            )
            turn1_preds, turn2_preds, comparison_preds = extract_multi_turn_yes_no(
                model_output, questions
            )

            result = {
                "attribute": task["attribute"],
                "desc": task["desc"],
                "comparison_type": "multi_turn",
                "jsonl_file": os.path.basename(task["jsonl_file"]),
                "line_idx": task["line_idx"],
                "questions": questions,
                "gt_answers": gt_answers,
                "turn1_pred": turn1_preds,
                "turn2_pred": turn2_preds,
                "comparison_questions": [
                    "Is the second image an improvement over the first image in terms of visual quality?",
                    "Is the second image more accurate to the prompt than the first image?",
                    "Does the second image show better composition than the first image?",
                ],
                "comparison_pred": comparison_preds,
                "model_output": model_output,
            }

            # Use lock if provided (for concurrent execution)
            if lock:
                with lock:
                    with open(
                        task["out_path_comparison"], "w", encoding="utf-8"
                    ) as fout:
                        json.dump(result, fout, ensure_ascii=False, indent=2)
            else:
                with open(task["out_path_comparison"], "w", encoding="utf-8") as fout:
                    json.dump(result, fout, ensure_ascii=False, indent=2)

        return True  # Success
    except Exception as e:
        print(f"[Error] {task.get('turn1_img_path', 'unknown')} : {e}")
        return False  # Failure


def concurrent_process_tasks(tasks, client, model, eval_turn, max_workers=4):
    """Process tasks concurrently using ThreadPoolExecutor"""
    lock = threading.Lock()
    failed_tasks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                process_single_task, task, client, model, eval_turn, lock
            ): task
            for task in tasks
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing evaluation tasks",
        ):
            task = future_to_task[future]
            try:
                success = future.result()
                if not success:
                    failed_tasks.append(task)
            except Exception as e:
                print(f"Task failed with exception: {e}")
                failed_tasks.append(task)

    return failed_tasks


def main(args):
    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    if args.multi_turn:
        tasks = collect_multi_turn_tasks(
            args.jsonl_dir,
            args.image_dir,
            args.eval_model,
            args.output_dir,
            args.sample_idx_file,
            args.postfix,
        )
    else:
        # Use original single-turn collection logic (not implemented here for brevity)
        raise NotImplementedError(
            "Single-turn evaluation not implemented in this script"
        )

    print(f"Total tasks to process: {len(tasks)}")

    # Use concurrent processing if max_workers > 1, otherwise use sequential processing
    if args.max_workers > 1:
        print(f"Using concurrent processing with {args.max_workers} workers")

        # Process tasks concurrently
        failed_tasks = concurrent_process_tasks(
            tasks, client, args.model, args.eval_turn, args.max_workers
        )

        # Retry failed tasks sequentially if any
        retry_count = 0
        max_retries = 3
        while failed_tasks and retry_count < max_retries:
            retry_count += 1
            print(
                f"Retrying {len(failed_tasks)} failed tasks (attempt {retry_count}/{max_retries})..."
            )
            time.sleep(5)

            # Retry failed tasks sequentially with more careful error handling
            new_failed_tasks = []
            for task in tqdm(failed_tasks, desc=f"Retry attempt {retry_count}"):
                try:
                    success = process_single_task(
                        task, client, args.model, args.eval_turn
                    )
                    if not success:
                        new_failed_tasks.append(task)
                        time.sleep(2)  # Add delay between retries
                except Exception as e:
                    print(
                        f"[Retry Error] {task.get('turn1_img_path', 'unknown')} : {e}"
                    )
                    new_failed_tasks.append(task)
                    time.sleep(2)

            failed_tasks = new_failed_tasks

        if failed_tasks:
            print(
                f"Warning: {len(failed_tasks)} tasks still failed after {max_retries} retry attempts"
            )
    else:
        print("Using sequential processing")

        # Original sequential processing with retry logic
        retry_tasks = []
        while tasks:
            retry_tasks.clear()
            for task in tqdm(tasks, desc="Processing tasks sequentially"):
                try:
                    success = process_single_task(
                        task, client, args.model, args.eval_turn
                    )
                    if not success:
                        retry_tasks.append(task)
                        time.sleep(2)
                except Exception as e:
                    print(f"[Error] {task.get('turn1_img_path', 'unknown')} : {e}")
                    retry_tasks.append(task)
                    time.sleep(2)

            if retry_tasks:
                print(f"Retrying {len(retry_tasks)} failed tasks...")
                time.sleep(5)
            tasks = retry_tasks.copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_dir", type=str, required=True, help="Directory containing jsonl files"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--eval_model", type=str, required=True, help="name of the eval model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output json files",
    )
    parser.add_argument("--api_key", type=str, default="sk-xxx", help="OpenAI API key")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base url",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--sample_idx_file",
        type=str,
        default=None,
        help="File containing sample indices",
    )
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument(
        "--multi_turn", action="store_true", help="Enable multi-turn evaluation"
    )
    parser.add_argument(
        "--eval_turn",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="Which turn to evaluate: 1 (first turn only), 2 (second turn only), both (comparison)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers for parallel processing (default: 4, set to 1 for sequential processing)",
    )

    args = parser.parse_args()
    main(args)
