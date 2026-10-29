from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rek.parity_core.workflow import EvolutionGraph

DEFAULT_LOG_ROOT = "YOUR_LOG_DIR_HERE"


def get_log_root(cli_value: str | None = None) -> str:
    return cli_value or os.getenv("TRAINER_LOG_ROOT") or DEFAULT_LOG_ROOT


def save_trace_file(instance_id: str, trace_data: list[dict[str, Any]], log_dir: str | None = None) -> None:
    """Save detailed reasoning traces as JSON."""
    try:
        if log_dir is None:
            log_dir = os.path.join(get_log_root(), "traces")
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{instance_id}.json")
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(trace_data, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Failed to save trace for {instance_id}: {exc}")


def setup_logger(log_dir: str | None = None) -> tuple[logging.Logger, str]:
    log_dir = get_log_root(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def load_jsonl(path: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def make_batches(data: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(data), batch_size):
        yield data[index : index + batch_size]


def filter_ground_truth(files: list[str]) -> list[str]:
    if not isinstance(files, list):
        return []
    allowed_exts = (".py", ".pyx", ".pxd", ".tp")
    return [
        file_path
        for file_path in files
        if "test" not in file_path and "conftest" not in file_path and file_path.endswith(allowed_exts)
    ]


def get_robust_ground_truth(sample: dict[str, Any]) -> list[str]:
    """Extract ground truth from explicit fields first, then fall back to patch parsing."""
    raw_gt: list[str] = []

    for field in ["files_preview", "gold_files_path", "files", "resolved"]:
        value = sample.get(field)
        if isinstance(value, list) and value:
            raw_gt = value
            break

    if not raw_gt and "patch" in sample:
        patch_text = sample.get("patch", "")
        matches = re.findall(r"^(?:--- a/|\+\+\+ b/)(.*)$", patch_text, re.MULTILINE)
        raw_gt = list({match.strip() for match in matches if match.strip() != "dev/null"})

    return raw_gt


def process_sample(sample: dict[str, Any], engine: EvolutionGraph) -> dict[str, Any]:
    issue = sample.get("problem_statement", "")
    base_commit = sample.get("base_commit", "HEAD")
    instance_id = sample.get("instance_id", "unknown")
    patch_text = sample.get("patch", "")

    raw_gt = get_robust_ground_truth(sample)
    ground_truth = filter_ground_truth(raw_gt)
    if not ground_truth or not patch_text:
        return {"ignored": True, "instance_id": instance_id, "reason": "No valid ground truth or patch."}

    distilled_json = engine.distill_knowledge_for_sample(issue, ground_truth, base_commit, patch_text)
    return {
        "instance_id": instance_id,
        "issue": issue,
        "distilled_data": distilled_json,
        "ignored": False,
    }


def run_training(
    repo_path: str,
    kb_path: str,
    elements_path: str,
    train_data_path: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    workers: int,
    log_root: str | None = None,
) -> dict[str, Any]:
    resolved_log_root = get_log_root(log_root)
    os.environ["TRAINER_LOG_ROOT"] = resolved_log_root
    logger, log_file = setup_logger(log_root)
    engine = EvolutionGraph(
        repo_path=repo_path,
        kb_path=kb_path,
        elements_path=elements_path,
        model_name=model_name,
    )

    base_kb_path = Path(kb_path)
    dataset = load_jsonl(train_data_path)
    logger.info("Loaded %s samples.", len(dataset))

    random.seed(42)
    random.shuffle(dataset)
    logger.info("Dataset successfully shuffled (seed=42).")

    trace_dir = os.path.join(get_log_root(log_root), f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(trace_dir, exist_ok=True)
    logger.info("Traces will be saved to: %s", trace_dir)

    epoch_outputs: list[str] = []
    for epoch in range(epochs):
        current_epoch = epoch + 1
        logger.info("=== Starting Epoch %s ===", current_epoch)

        epoch_kb_name = f"{base_kb_path.stem}_epoch_{current_epoch}{base_kb_path.suffix}"
        epoch_kb_path = base_kb_path.parent / epoch_kb_name
        engine.graph_db.set_storage_path(str(epoch_kb_path))
        logger.info("Initializing epoch KB file: %s", epoch_kb_path)
        engine.graph_db.save_state()
        epoch_outputs.append(str(epoch_kb_path))

        batches = list(make_batches(dataset, batch_size))
        pbar = tqdm(total=len(dataset), desc=f"Epoch {current_epoch}", unit="sample")

        for batch in batches:
            batch_failures: list[dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_sample, sample, engine): sample for sample in batch}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result and not result.get("ignored"):
                            batch_failures.append(result)
                            logger.info(
                                "PR %s distilled %s knowledge items.",
                                result["instance_id"],
                                len(result.get("distilled_data", [])),
                            )
                    except Exception as exc:
                        logger.error("Worker error: %s", exc)
                    pbar.update(1)

            if batch_failures:
                engine.run_fast_injection(batch_failures)

        pbar.close()
        logger.info("Epoch %s completed. Final state saved to %s", current_epoch, epoch_kb_path)

    return {
        "log_file": log_file,
        "trace_dir": trace_dir,
        "epoch_outputs": epoch_outputs,
        "sample_count": len(dataset),
    }
