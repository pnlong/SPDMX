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


def _patch_torch_load_compat(train_py: Path) -> bool:
    """Make Lightning resume work on PyTorch 2.6+ (weights_only default)."""
    text = train_py.read_text(encoding="utf-8")
    changed = False
    if 'kwargs.setdefault("weights_only", False)' in text:
        text = text.replace(
            'kwargs.setdefault("weights_only", False)',
            'kwargs["weights_only"] = False  # Lightning may pass True; full ckpts need False',
        )
        changed = True
    if "_torch_load_checkpoint" in text and "add_safe_globals([TaskManager])" in text:
        if changed:
            train_py.write_text(text, encoding="utf-8")
        return changed
    needle = "from utils.utils import str2bool\n"
    if needle not in text:
        raise SystemExit(f"could not patch torch.load compat into {train_py}")
    insert = needle + '''
# PyTorch 2.6+ defaults torch.load(weights_only=True). Lightning resume pickles
# TaskManager inside last.ckpt — allowlist it and default to full loads.
try:
    torch.serialization.add_safe_globals([TaskManager])
except Exception:
    pass
_torch_load = torch.load


def _torch_load_checkpoint(*args, **kwargs):
    # Lightning may pass weights_only=True; full trainer ckpts need False.
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_checkpoint
'''
    train_py.write_text(text.replace(needle, insert, 1), encoding="utf-8")
    return True


def _patch_torch_load(path: Path) -> bool:
    """PyTorch 2.6 defaults torch.load to weights_only=True; YourMT3 ckpts pickle TaskManager."""
    text = path.read_text(encoding="utf-8")
    old = 'torch.load(dir_info["last_ckpt_path"])'
    new = 'torch.load(dir_info["last_ckpt_path"], weights_only=False)'
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


_STEP_BAR = '''
class StepProgressBar(TQDMProgressBar):
    """Show global optimizer steps, not Lightning's per-epoch counter."""

    def _step_total(self, trainer) -> int:
        total = getattr(trainer, "max_steps", None)
        try:
            total = int(total)
        except (TypeError, ValueError):
            return -1
        return total

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        total = self._step_total(trainer)
        if total > 0:
            self.train_progress_bar.total = total
            self.train_progress_bar.n = trainer.global_step
            self.train_progress_bar.set_description(f"Step {trainer.global_step}/{total}")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        total = self._step_total(trainer)
        if total > 0:
            self.train_progress_bar.set_description(f"Step {trainer.global_step}/{total}")
            return
        super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        total = self._step_total(trainer)
        if total > 0:
            n = min(int(trainer.global_step), total)
            if self._should_update(n, total):
                self.train_progress_bar.n = n
                self.train_progress_bar.set_description(f"Step {n}/{total}")
                self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            return
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


'''


def _patch_step_progress(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "class StepProgressBar" in text and "StepProgressBar()" in text:
        return False
    if "from pytorch_lightning.callbacks import TQDMProgressBar" not in text:
        text = text.replace(
            "from pytorch_lightning.callbacks import LearningRateMonitor\n",
            "from pytorch_lightning.callbacks import LearningRateMonitor\n"
            "from pytorch_lightning.callbacks import TQDMProgressBar\n",
            1,
        )
    if "class StepProgressBar" not in text:
        needle = "def initialize_trainer("
        if needle not in text:
            raise SystemExit(f"could not insert StepProgressBar in {path}")
        text = text.replace(needle, _STEP_BAR + needle, 1)
    text = text.replace(
        "callbacks=[checkpoint_callback, lr_monitor],",
        "callbacks=[checkpoint_callback, lr_monitor, StepProgressBar()],",
    )
    path.write_text(text, encoding="utf-8")
    return True


def _patch_limit_val_batches(train_py: Path, init_train: Path) -> bool:
    """Add --limit-val-batches CLI and wire it into Lightning TRAINER config."""
    changed = False
    train_text = train_py.read_text(encoding="utf-8")
    if "--limit-val-batches" not in train_text:
        lines = train_text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if (not inserted) and "'--val-interval'" in line and "add_argument" in line:
                out.append(
                    "parser.add_argument('-lvb', '--limit-val-batches', type=int, default=None, "
                    "help='cap validation to this many batches (Lightning limit_val_batches). "
                    "If None, use config (full val set).')\n"
                )
                inserted = True
        if not inserted:
            raise SystemExit(f"could not find val-interval arg in {train_py}")
        train_py.write_text("".join(out), encoding="utf-8")
        changed = True

    init_text = init_train.read_text(encoding="utf-8")
    marker = 'getattr(args, "limit_val_batches", None)'
    if marker not in init_text:
        block = '''    if stage == 'train' and args.val_interval is not None:
        shared_cfg["TRAINER"]["check_val_every_n_epoch"] = None
        shared_cfg["TRAINER"]["val_check_interval"] = int(args.val_interval)
'''
        insert = block + '''    if stage == 'train' and getattr(args, "limit_val_batches", None) is not None:
        shared_cfg["TRAINER"]["limit_val_batches"] = int(args.limit_val_batches)
'''
        if block not in init_text:
            raise SystemExit(f"could not patch limit_val_batches into {init_train}")
        init_train.write_text(init_text.replace(block, insert, 1), encoding="utf-8")
        changed = True
    return changed


def _patch_auto_resume(train_py: Path, init_train: Path) -> bool:
    """Full Lightning resume via ckpt_path + periodic last.ckpt saves on step."""
    changed = False
    train_text = train_py.read_text(encoding="utf-8")
    broken = '''    # last_ckpt_path can be None
    if dir_info["last_ckpt_path"] is not None:
        checkpoint = torch.load(dir_info["last_ckpt_path"], weights_only=False)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)
'''
    broken_old = '''    # last_ckpt_path can be None
    if dir_info["last_ckpt_path"] is not None:
        checkpoint = torch.load(dir_info["last_ckpt_path"])
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)
'''
    fixed = '''    # Auto-resume: same project/exp_id → amt/logs/<project>/<exp_id>/checkpoints/last.ckpt
    # Pass ckpt_path so Lightning restores optimizer, scheduler, and global_step.
    trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)
'''
    if "Auto-resume: same project/exp_id" not in train_text:
        if broken in train_text:
            train_py.write_text(train_text.replace(broken, fixed, 1), encoding="utf-8")
            changed = True
        elif broken_old in train_text:
            train_py.write_text(train_text.replace(broken_old, fixed, 1), encoding="utf-8")
            changed = True

    init_text = init_train.read_text(encoding="utf-8")
    if "every_n_train_steps" not in init_text:
        old_block = '''    if stage == 'train' and args.val_interval is not None:
        shared_cfg["TRAINER"]["check_val_every_n_epoch"] = None
        shared_cfg["TRAINER"]["val_check_interval"] = int(args.val_interval)
'''
        new_block = '''    if stage == 'train' and args.val_interval is not None:
        shared_cfg["TRAINER"]["check_val_every_n_epoch"] = None
        shared_cfg["TRAINER"]["val_check_interval"] = int(args.val_interval)
        shared_cfg["CHECKPOINT"]["every_n_train_steps"] = int(args.val_interval)
        shared_cfg["CHECKPOINT"]["save_on_train_epoch_end"] = False
'''
        if old_block in init_text:
            init_text = init_text.replace(old_block, new_block, 1)
            early = (
                '    # define checkpoint callback\n'
                '    checkpoint_callback = ModelCheckpoint(**shared_cfg["CHECKPOINT"],)\n\n'
            )
            if early in init_text and init_text.find(early) < init_text.find("every_n_train_steps"):
                init_text = init_text.replace(early, "", 1)
                marker = 'shared_cfg["CHECKPOINT"]["save_on_train_epoch_end"] = False\n'
                init_text = init_text.replace(
                    marker,
                    marker
                    + "\n    # define checkpoint callback (after CHECKPOINT overrides above)\n"
                    + '    checkpoint_callback = ModelCheckpoint(**shared_cfg["CHECKPOINT"],)\n',
                    1,
                )
            init_train.write_text(init_text, encoding="utf-8")
            changed = True
    return changed


def _patch_test_subbsz_and_per_track(ymt3_py: Path) -> bool:
    """Faster test inference sub-batch + per-track JSONL for bootstrap CIs."""
    text = ymt3_py.read_text(encoding="utf-8")
    changed = False

    # Only patch test_step (validation_step has a near-identical prefix).
    test_old = '''    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:
        # File-wise evaluation
        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3'''
    test_new = '''    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:
        # File-wise evaluation
        # Prefer test BSZ (larger) for GPU throughput; override via SPDMX_TEST_SUBBSZ.
        import os as _os
        _env_bsz = _os.environ.get("SPDMX_TEST_SUBBSZ")
        _base_bsz = int(_env_bsz) if _env_bsz else int(
            self.shared_cfg["BSZ"].get("test", self.shared_cfg["BSZ"]["validation"]))
        if self.task_manager.num_decoding_channels == 1:
            bsz = _base_bsz
        else:
            bsz = max(1, _base_bsz // self.task_manager.num_decoding_channels * 3)'''
    if "def test_step" in text and "SPDMX_TEST_SUBBSZ" not in text.split("def test_step")[1].split("def on_test_epoch_end")[0]:
        if test_old in text:
            text = text.replace(test_old, test_new, 1)
            changed = True
        elif "Prefer test BSZ (larger) for GPU throughput; override via SPDMX_TEST_SUBBSZ." in text.split("def test_step")[1].split("def on_test_epoch_end")[0]:
            pass  # already patched in test_step
        else:
            # test_step still on validation BSZ; force-replace that block only under test_step
            marker = "    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:\n"
            idx = text.find(marker)
            if idx < 0:
                raise SystemExit(f"could not find test_step in {ymt3_py}")
            end = text.find("\n    def ", idx + len(marker))
            block = text[idx:end]
            old_inner = '''        # File-wise evaluation
        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3'''
            new_inner = '''        # File-wise evaluation
        # Prefer test BSZ (larger) for GPU throughput; override via SPDMX_TEST_SUBBSZ.
        import os as _os
        _env_bsz = _os.environ.get("SPDMX_TEST_SUBBSZ")
        _base_bsz = int(_env_bsz) if _env_bsz else int(
            self.shared_cfg["BSZ"].get("test", self.shared_cfg["BSZ"]["validation"]))
        if self.task_manager.num_decoding_channels == 1:
            bsz = _base_bsz
        else:
            bsz = max(1, _base_bsz // self.task_manager.num_decoding_channels * 3)'''
            if old_inner not in block:
                raise SystemExit(f"could not patch test_step subbsz in {ymt3_py}")
            text = text[:idx] + block.replace(old_inner, new_inner, 1) + text[end:]
            changed = True

    dump_marker = "# SPDMX_PER_TRACK_DUMP"
    if dump_marker not in text:
        anchor = '''            self.test_metrics[dataloader_idx].bulk_update(drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(non_drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(instr_metric)'''
        dump = '''            self.test_metrics[dataloader_idx].bulk_update(drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(non_drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(instr_metric)
            # SPDMX_PER_TRACK_DUMP
            try:
                import json as _json
                import math as _math
                import os as _os
                _ids = [notes_dict[k] for k in notes_dict.keys() if k.endswith("_id")]
                _track = str(_ids[0]) if _ids else f"batch_{batch_idx}"
                def _f(x):
                    try:
                        v = float(x)
                    except Exception:
                        return None
                    return None if _math.isnan(v) else v
                _row = {
                    "track_id": _track,
                    "onset_f": _f(non_drum_metric.get("onset_f")),
                    "offset_f": _f(non_drum_metric.get("offset_f")),
                    "multi_f": _f(instr_metric.get("multi_f")),
                    "rank": int(getattr(self, "global_rank", 0)),
                }
                _dir = _os.environ.get("SPDMX_PER_TRACK_DIR")
                if _dir:
                    _os.makedirs(_dir, exist_ok=True)
                    _path = _os.path.join(_dir, f"per_track_rank{getattr(self, 'global_rank', 0)}.jsonl")
                    with open(_path, "a") as _fh:
                        _fh.write(_json.dumps(_row) + "\\n")
            except Exception as _exc:
                print(f"[SPDMX] per-track dump failed: {_exc}")'''
        if anchor not in text:
            raise SystemExit(f"could not find test_step bulk_update anchor in {ymt3_py}")
        text = text.replace(anchor, dump, 1)
        changed = True

    if changed:
        ymt3_py.write_text(text, encoding="utf-8")
    return changed


def _patch_config_dataloader(config_py: Path) -> bool:
    text = config_py.read_text(encoding="utf-8")
    marker = "# SPDMX_DATAIO_ENV"
    if marker in text:
        return False
    old = '''    "DATAIO": { # do not set `shuffle` here. 
        "num_workers": 4, # num_worker is per GPU in DDP mode
        "prefetch_factor": 2, #2,
        "pin_memory": True,
        "persistent_workers": False,
    },'''
    new = '''    "DATAIO": { # do not set `shuffle` here.  # SPDMX_DATAIO_ENV
        "num_workers": int(__import__("os").environ.get("SPDMX_TEST_NUM_WORKERS", "4")),
        "prefetch_factor": int(__import__("os").environ.get("SPDMX_TEST_PREFETCH", "2")),
        "pin_memory": True,
        "persistent_workers": __import__("os").environ.get("SPDMX_TEST_PERSISTENT_WORKERS", "0") == "1",
    },'''
    if old not in text:
        # Try compact form
        old2 = '''    "DATAIO": { # do not set `shuffle` here.
        "num_workers": 4, # num_worker is per GPU in DDP mode
        "prefetch_factor": 2, #2,
        "pin_memory": True,
        "persistent_workers": False,
    },'''
        if old2 not in text:
            return False
        text = text.replace(old2, new, 1)
    else:
        text = text.replace(old, new, 1)
    config_py.write_text(text, encoding="utf-8")
    return True


OVERLAY_DIR = Path(__file__).resolve().parent / "yourmt3_overlays"


def _install_overlays(yourmt3_src: Path) -> list[str]:
    """Copy tracked overlays into the gitignored YourMT3 clone (packing + fast eval)."""
    mapping = {
        "datasets_eval.py": yourmt3_src / "utils" / "datasets_eval.py",
        "ymt3.py": yourmt3_src / "model" / "ymt3.py",
        "test.py": yourmt3_src / "test.py",
    }
    installed: list[str] = []
    for name, dest in mapping.items():
        src = OVERLAY_DIR / name
        if not src.is_file():
            if name == "test.py":
                print(f"[SPDMX] overlay {name}: not in repo yet (optional; skipping)")
                continue
            raise SystemExit(f"missing overlay {src}")
        if not dest.parent.is_dir():
            raise SystemExit(f"YourMT3 path missing: {dest.parent}")
        new = src.read_bytes()
        old = dest.read_bytes() if dest.is_file() else None
        if old == new:
            # Still report packing status for the critical file
            if name == "datasets_eval.py":
                text = new.decode("utf-8", errors="replace")
                kind = "best-fit" if "best-fit" in text or "First-fit decreasing" in text else "OLD-greedy"
                print(f"[SPDMX] overlay {name}: already current ({kind})")
            continue
        dest.write_bytes(new)
        text = new.decode("utf-8", errors="replace")
        kind = ""
        if name == "datasets_eval.py":
            kind = " best-fit" if ("best-fit" in text or "First-fit decreasing" in text) else " OLD-greedy?"
        installed.append(str(dest) + f" (overlay {name}{kind})")
        print(f"[SPDMX] overlay {name}: installed{kind} → {dest}")
    return installed


def _patch_test_auto_overlay(test_py: Path) -> bool:
    """Ensure test.py applies overlays before importing data modules."""
    text = test_py.read_text(encoding="utf-8")
    new_marker = "yourmt3_overlays"
    bootstrap = '''""" test.py """
# SPDMX_AUTO_OVERLAY: copy tracked packing/eval overlays before importing YourMT3 modules.
# (YourMT3/ is gitignored; `git pull` alone does not update these files.)
def _spdmx_auto_overlay() -> None:
    try:
        from pathlib import Path

        src_dir = Path(__file__).resolve().parent  # .../YourMT3/amt/src
        # .../experiments/transcription
        trans_dir = src_dir.parents[3]
        overlay_dir = trans_dir / "yourmt3_overlays"
        if not overlay_dir.is_dir():
            print(f"[SPDMX] auto-overlay: missing {overlay_dir}")
            return
        mapping = {
            "datasets_eval.py": src_dir / "utils" / "datasets_eval.py",
            "ymt3.py": src_dir / "model" / "ymt3.py",
            # Do not rewrite this running test.py from here.
        }
        for name, dest in mapping.items():
            src = overlay_dir / name
            if not src.is_file():
                print(f"[SPDMX] auto-overlay: missing {src}")
                continue
            new = src.read_bytes()
            old = dest.read_bytes() if dest.is_file() else None
            if old == new:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(new)
            print(f"[SPDMX] auto-overlay: installed {name} → {dest}")
    except Exception as exc:  # noqa: BLE001 — never block stock YourMT3 test
        print(f"[SPDMX] auto-overlay skipped: {exc}")


_spdmx_auto_overlay()

import os
import pprint
import argparse
import torch

from utils.data_modules import AMTDataModule'''
    if "SPDMX_AUTO_OVERLAY" in text and new_marker in text and "experiments.transcription.patch_yourmt3" not in text:
        return False
    # Replace from docstring through first data_modules import.
    start = text.find('""" test.py """')
    if start < 0:
        return False
    end = text.find("from utils.data_modules import AMTDataModule", start)
    if end < 0:
        return False
    text = text[:start] + bootstrap + text[end + len("from utils.data_modules import AMTDataModule"):]
    # bootstrap already ends with that import line
    test_py.write_text(text, encoding="utf-8")
    return True


def apply_yourmt3_patches(yourmt3_src: Path | None = None) -> list[str]:
    """Apply local YourMT3 fixes. Returns list of patched file paths."""
    src = Path(yourmt3_src) if yourmt3_src is not None else YOURMT3_SRC
    if not src.is_dir():
        raise SystemExit(f"YourMT3 src missing: {src}")
    patched: list[str] = []
    init_train = src / "model" / "init_train.py"
    train_py = src / "train.py"
    test_py = src / "test.py"
    ymt3_py = src / "model" / "ymt3.py"
    config_py = src / "config" / "config.py"
    # Install packing / skip-tokenize / per-track overlays first (authoritative).
    patched.extend(_install_overlays(src))
    if test_py.is_file() and _patch_test_auto_overlay(test_py):
        patched.append(str(test_py) + " (auto-overlay)")
    if init_train.is_file() and _patch_init_train(init_train):
        patched.append(str(init_train))
    if init_train.is_file() and _patch_step_progress(init_train):
        patched.append(str(init_train) + " (step bar)")
    if train_py.is_file() and _patch_train_py(train_py):
        patched.append(str(train_py))
    if train_py.is_file() and _patch_torch_load_compat(train_py):
        patched.append(str(train_py) + " (torch.load compat)")
    for ckpt_py in (train_py, test_py):
        if ckpt_py.is_file() and _patch_torch_load(ckpt_py):
            patched.append(str(ckpt_py))
    if train_py.is_file() and init_train.is_file() and _patch_limit_val_batches(train_py, init_train):
        patched.append(str(train_py) + " (limit-val-batches)")
    if train_py.is_file() and init_train.is_file() and _patch_auto_resume(train_py, init_train):
        patched.append(str(train_py) + " (auto-resume)")
    # Overlays already include per-track + subbsz + packing; keep surgical patch for older trees.
    if ymt3_py.is_file() and "__packed__" not in ymt3_py.read_text(encoding="utf-8"):
        if _patch_test_subbsz_and_per_track(ymt3_py):
            patched.append(str(ymt3_py) + " (per-track + test subbsz)")
    if config_py.is_file() and _patch_config_dataloader(config_py):
        patched.append(str(config_py) + " (dataloader env)")
    return patched
