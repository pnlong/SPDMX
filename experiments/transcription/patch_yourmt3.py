"""Idempotent patches for the local (gitignored) YourMT3 clone.

YourMT3 is cloned under experiments/transcription/YourMT3/ and is not in git,
so collaborator machines do not pick up hand-edits. Call ``apply_yourmt3_patches``
from setup and before train.
"""

from __future__ import annotations

from pathlib import Path

from experiments.transcription.paths import TRANS_DIR

YOURMT3_SRC = TRANS_DIR / "YourMT3" / "amt" / "src"


def _patch_init_train(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "wandb_mode == \"disabled\"" in text or "wandb_mode == 'disabled'" in text:
        return False

    old_logger_block = '''    # wandb_logger = WandbLogger(log_model="all",
    #                            project=args.project,
    #                            id=args.exp_id,
    #                            allow_val_change=True,
    #                            **shared_cfg['WANDB'])
    wandb_logger = None
'''
    new_logger_block = '''    wandb_mode = str(shared_cfg["WANDB"].get("mode", "online"))
    if wandb_mode == "disabled":
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(
            log_model="all",
            project=args.project,
            id=args.exp_id,
            allow_val_change=True,
            **shared_cfg["WANDB"],
        )
'''
    if old_logger_block in text:
        text = text.replace(old_logger_block, new_logger_block, 1)
    elif "wandb_logger = None" in text and "WandbLogger(" not in text.split("wandb_logger = None")[0][-400:]:
        # Broader fallback: someone forced None after the WANDB cache_dir block.
        marker = '        del shared_cfg["WANDB"]["cache_dir"]  # remove cache_dir from shared_cfg\n'
        if marker not in text:
            marker = "        del shared_cfg[\"WANDB\"][\"cache_dir\"]  # remove cache_dir from shared_cfg\n"
        # Find wandb_logger = None after create logger section
        idx = text.find("wandb_logger = None")
        if idx < 0:
            raise SystemExit(f"could not patch WandbLogger in {path}")
        # Replace from any commented WandbLogger block through wandb_logger = None
        start = text.rfind("# wandb_logger", 0, idx)
        if start < 0:
            start = idx
        end = idx + len("wandb_logger = None\n")
        text = text[:start] + new_logger_block + text[end:]
    else:
        # Already has WandbLogger(...) — ensure trainer uses it
        pass

    text = text.replace(
        "#                        logger=wandb_logger,",
        "                        logger=wandb_logger,",
    )
    text = text.replace(
        "                        # logger=wandb_logger,",
        "                        logger=wandb_logger,",
    )
    # Uncomment the post-trainer wandb config update if still commented
    commented = '''    # # Update wandb logger (for DDP)
    # if trainer.global_rank == 0:
    #     wandb_logger.experiment.config.update(args, allow_val_change=True)
'''
    uncommented = '''    # Update wandb logger (for DDP)
    if trainer.global_rank == 0 and wandb_logger is not None:
        wandb_logger.experiment.config.update(args, allow_val_change=True)
'''
    if commented in text:
        text = text.replace(commented, uncommented, 1)
    elif (
        "wandb_logger.experiment.config.update(args, allow_val_change=True)" in text
        and "and wandb_logger is not None" not in text
    ):
        text = text.replace(
            "if trainer.global_rank == 0:\n        wandb_logger.experiment.config.update(args, allow_val_change=True)",
            "if trainer.global_rank == 0 and wandb_logger is not None:\n        wandb_logger.experiment.config.update(args, allow_val_change=True)",
            1,
        )

    path.write_text(text, encoding="utf-8")
    return True


def _patch_train_py(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    old = '''    # Logging config updated by args
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update({"audio_cfg": model.audio_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update({"model_cfg": model.model_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update(model.shared_cfg, allow_val_change=True)

    wandb_logger.watch(model, log='gradients', log_freq=5000)
'''
    new = '''    # Logging config updated by args
    if trainer.global_rank == 0 and wandb_logger is not None:
        wandb_logger.experiment.config.update({"audio_cfg": model.audio_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update({"model_cfg": model.model_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update(model.shared_cfg, allow_val_change=True)
        wandb_logger.watch(model, log='gradients', log_freq=5000)
'''
    if "and wandb_logger is not None" in text and "wandb_logger.watch" in text.split("and wandb_logger is not None", 1)[1][:400]:
        return False
    if old in text:
        text = text.replace(old, new, 1)
    else:
        # Minimal surgical guard if formatting differs
        text2 = text.replace(
            "if trainer.global_rank == 0:\n        wandb_logger.experiment.config.update({\"audio_cfg\"",
            "if trainer.global_rank == 0 and wandb_logger is not None:\n        wandb_logger.experiment.config.update({\"audio_cfg\"",
            1,
        )
        if text2 == text:
            raise SystemExit(f"could not patch wandb guards in {path}")
        text = text2
        # Indent watch under the same guard if still bare
        if "\n    wandb_logger.watch(model" in text:
            text = text.replace(
                "\n    wandb_logger.watch(model, log='gradients', log_freq=5000)\n",
                "\n        wandb_logger.watch(model, log='gradients', log_freq=5000)\n",
                1,
            )
    path.write_text(text, encoding="utf-8")
    return True


def apply_yourmt3_patches(yourmt3_src: Path | None = None) -> list[str]:
    """Apply local YourMT3 fixes. Returns list of patched file paths."""
    src = Path(yourmt3_src) if yourmt3_src is not None else YOURMT3_SRC
    if not src.is_dir():
        raise SystemExit(f"YourMT3 src missing: {src}")
    patched: list[str] = []
    init_train = src / "model" / "init_train.py"
    train_py = src / "train.py"
    if init_train.is_file() and _patch_init_train(init_train):
        patched.append(str(init_train))
    if train_py.is_file() and _patch_train_py(train_py):
        patched.append(str(train_py))
    return patched
