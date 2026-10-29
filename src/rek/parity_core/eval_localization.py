from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from tqdm import tqdm

from rek.parity_core.git_time_travel import ASTGroundTruthExtractor
from rek.parity_core.workflow import EvolutionGraph

logger = logging.getLogger("EvalLocalization")


def setup_detailed_logger(log_dir: str = "logs/eval") -> tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_{timestamp}.log")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - \n%(message)s\n" + "-" * 50)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    engine_logger = logging.getLogger("EvolutionEngine")
    engine_logger.handlers.clear()
    engine_logger.addHandler(file_handler)
    engine_logger.addHandler(console_handler)
    engine_logger.setLevel(logging.INFO)

    return logger, log_file


def calculate_metrics(preds: set[str], gts: set[str]) -> tuple[float, float, float]:
    if not preds and not gts:
        return 1.0, 1.0, 1.0
    if not preds or not gts:
        return 0.0, 0.0, 0.0
    hits = preds.intersection(gts)
    precision = len(hits) / len(preds) if preds else 0.0
    recall = len(hits) / len(gts) if gts else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def parse_swebench_gt(edit_functions: list[str], added_functions: list[str]) -> tuple[list[str], list[str], list[str]]:
    gt_files = set()
    gt_modules = set()
    gt_functions = set()

    all_functions = set(edit_functions + added_functions)
    for item in all_functions:
        if ":" not in item:
            continue
        file_path, element = item.split(":", 1)
        gt_files.add(file_path)
        if "." in element:
            module_name, function_name = element.rsplit(".", 1)
            gt_modules.add(f"{file_path}::{module_name}")
            gt_functions.add(f"{file_path}::{module_name}.{function_name}")
        else:
            gt_modules.add(f"{file_path}::Global")
            gt_functions.add(f"{file_path}::{element}")

    return list(gt_files), list(gt_modules), list(gt_functions)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_localization_eval(
    repo_path: str,
    kb_path: str,
    elements_path: str,
    test_data_path: str,
    model_name: str,
    instance_id: str | None = None,
    swebench_gt_path: str | None = None,
    log_dir: str = "logs/eval",
) -> dict[str, Any]:
    logger, log_path = setup_detailed_logger(log_dir)
    logger.info("Starting advanced localization eval | Model: %s", model_name)

    engine = EvolutionGraph(
        repo_path=repo_path,
        kb_path=kb_path,
        elements_path=elements_path,
        model_name=model_name,
    )
    gt_extractor = ASTGroundTruthExtractor(repo_path)

    swebench_gt_cache: dict[str, dict[str, Any]] = {}
    if swebench_gt_path and os.path.exists(swebench_gt_path):
        logger.info("Loading SWE-bench GT cache from: %s", swebench_gt_path)
        for item in load_jsonl(swebench_gt_path):
            current_instance_id = item.get("instance_id")
            edit_functions = item.get("edit_functions", [])
            added_functions = item.get("added_functions", [])
            if current_instance_id and (edit_functions or added_functions):
                swebench_gt_cache[current_instance_id] = {
                    "edit_functions": edit_functions,
                    "added_functions": added_functions,
                }
        logger.info("Successfully cached %s official GT records.", len(swebench_gt_cache))
    elif swebench_gt_path:
        logger.warning("SWE-bench GT file not found at %s. Falling back to AST extraction.", swebench_gt_path)

    all_data = load_jsonl(test_data_path)
    if instance_id:
        all_data = [item for item in all_data if item.get("instance_id") == instance_id]
    if not all_data:
        raise RuntimeError("No matching instances found.")

    global_metrics = {
        "file": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "module": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "func": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }
    valid_eval_count = 0

    for sample in tqdm(all_data, desc="Localizing"):
        current_instance_id = sample.get("instance_id", "unknown")
        issue = sample.get("problem_statement", "")
        base_commit = sample.get("base_commit", "HEAD")
        patch_text = sample.get("patch", "")

        logger.info("\n==================================================")
        logger.info("[Processing] Instance: %s", current_instance_id)

        swe_cache = swebench_gt_cache.get(current_instance_id)
        if swe_cache and (swe_cache["edit_functions"] or swe_cache["added_functions"]):
            logger.info("[GT Source] Extracting from SWE-bench official annotations")
            gt_files, gt_modules, gt_functions = parse_swebench_gt(
                swe_cache["edit_functions"],
                swe_cache["added_functions"],
            )
            ast_files, _, _ = gt_extractor.extract_3level_gt(patch_text, base_commit)
            gt_files = list(set(gt_files).union(set(ast_files)))
        else:
            logger.info("[GT Source] Falling back to dynamic AST patch extraction")
            gt_files, gt_modules, gt_functions = gt_extractor.extract_3level_gt(patch_text, base_commit)

        logger.info("GT Files: %s", gt_files)
        logger.info("GT Modules: %s", gt_modules)
        logger.info("GT Functions: %s", gt_functions)

        try:
            result = engine.run_inference_for_single_sample(issue, gt_files, base_commit)
            if not result:
                continue

            raw_pred_details = result.get("candidates_details", [])
            pred_files: set[str] = set()
            pred_modules: set[str] = set()
            pred_functions: set[str] = set()
            for detail in raw_pred_details:
                historic_name = detail.get("file", "")
                element = detail.get("element", "Entire File")
                if historic_name == "NOT_FOUND" or not historic_name:
                    continue
                pred_files.add(historic_name)
                clean_element = element.split("(")[0].strip()
                if "." in clean_element:
                    module_name, function_name = clean_element.rsplit(".", 1)
                    pred_modules.add(f"{historic_name}::{module_name}")
                    pred_functions.add(f"{historic_name}::{module_name}.{function_name}")
                else:
                    pred_modules.add(f"{historic_name}::Global")
                    if clean_element and clean_element != "Entire File":
                        pred_functions.add(f"{historic_name}::{clean_element}")

            p_file, r_file, f1_file = calculate_metrics(pred_files, set(gt_files))
            p_mod, r_mod, f1_mod = calculate_metrics(pred_modules, set(gt_modules))
            p_func, r_func, f1_func = calculate_metrics(pred_functions, set(gt_functions))

            global_metrics["file"]["p"] += p_file
            global_metrics["file"]["r"] += r_file
            global_metrics["file"]["f1"] += f1_file
            global_metrics["module"]["p"] += p_mod
            global_metrics["module"]["r"] += r_mod
            global_metrics["module"]["f1"] += f1_mod
            global_metrics["func"]["p"] += p_func
            global_metrics["func"]["r"] += r_func
            global_metrics["func"]["f1"] += f1_func
            valid_eval_count += 1

            logger.info("[FINAL MATCH REPORT: %s]", current_instance_id)
            logger.info("[FILE LEVEL]     P: %.2f | R: %.2f | F1: %.2f", p_file, r_file, f1_file)
            logger.info("[MODULE LEVEL]   P: %.2f | R: %.2f | F1: %.2f", p_mod, r_mod, f1_mod)
            logger.info("[FUNCTION LEVEL] P: %.2f | R: %.2f | F1: %.2f", p_func, r_func, f1_func)
        except Exception as exc:
            logger.error("Error processing %s: %s", current_instance_id, exc, exc_info=True)

    metrics: dict[str, float] = {}
    if valid_eval_count > 0:
        metrics = {
            "file_precision": global_metrics["file"]["p"] / valid_eval_count,
            "file_recall": global_metrics["file"]["r"] / valid_eval_count,
            "file_f1": global_metrics["file"]["f1"] / valid_eval_count,
            "module_precision": global_metrics["module"]["p"] / valid_eval_count,
            "module_recall": global_metrics["module"]["r"] / valid_eval_count,
            "module_f1": global_metrics["module"]["f1"] / valid_eval_count,
            "function_precision": global_metrics["func"]["p"] / valid_eval_count,
            "function_recall": global_metrics["func"]["r"] / valid_eval_count,
            "function_f1": global_metrics["func"]["f1"] / valid_eval_count,
            "evaluated_samples": float(valid_eval_count),
        }
    else:
        logger.warning("No instances were successfully evaluated.")

    return {
        "metrics": metrics,
        "log_path": log_path,
        "evaluated_samples": valid_eval_count,
    }
