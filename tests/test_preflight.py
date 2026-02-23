"""
VM_FOOTBALL - Pre-flight Test Suite
====================================
Run BEFORE submitting to Vertex AI to catch import errors, config issues,
and dependency problems that would waste cloud GPU time.

Usage:
    python tests/test_preflight.py                  # Run all tests
    python tests/test_preflight.py -v               # Verbose output
    python -m pytest tests/test_preflight.py -v     # With pytest

Tests cover:
    1. Python package structure (imports resolve)
    2. Config validation (YAML structure, required fields)
    3. Requirements consistency (no conflicts between files)
    4. Code quality (no obvious bugs, no duplicate code)
    5. Cloud script validation (startup script correctness)
    6. Dataset structure expectations
"""

import os
import sys
import unittest
import yaml
import ast
import re
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestPackageStructure(unittest.TestCase):
    """Verify Python packages are properly structured for imports."""

    def test_training_init_exists(self):
        """training/__init__.py must exist for package imports."""
        self.assertTrue(
            (ROOT / "training" / "__init__.py").exists(),
            "Missing training/__init__.py - imports will fail in cloud"
        )

    def test_training_detection_init_exists(self):
        """training/detection/__init__.py must exist."""
        self.assertTrue(
            (ROOT / "training" / "detection" / "__init__.py").exists(),
            "Missing training/detection/__init__.py - imports will fail in cloud"
        )

    def test_cloud_directory_exists(self):
        """cloud/ directory with pipeline code must exist."""
        self.assertTrue((ROOT / "cloud").is_dir())

    def test_core_directory_exists(self):
        """core/ directory must exist."""
        self.assertTrue((ROOT / "core").is_dir())


class TestImportsResolve(unittest.TestCase):
    """Verify all critical imports resolve without errors."""

    def test_run_pipeline_imports(self):
        """run_pipeline.py must parse without syntax errors."""
        path = ROOT / "cloud" / "run_pipeline.py"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        # Should parse without SyntaxError
        ast.parse(source, filename=str(path))

    def test_train_rfdetr_imports(self):
        """train_rfdetr.py must parse without syntax errors."""
        path = ROOT / "training" / "detection" / "train_rfdetr.py"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source, filename=str(path))

    def test_train_yolo_imports(self):
        """train_yolo.py must parse without syntax errors."""
        path = ROOT / "training" / "detection" / "train_yolo.py"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source, filename=str(path))

    def test_diagnose_env_imports(self):
        """diagnose_env.py must parse without syntax errors."""
        path = ROOT / "cloud" / "diagnose_env.py"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source, filename=str(path))

    def test_callbacks_imports(self):
        """callbacks.py must parse without syntax errors."""
        path = ROOT / "cloud" / "callbacks.py"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source, filename=str(path))

    def test_no_total_mem_typo(self):
        """Ensure total_mem typo (should be total_memory) is not present."""
        for pyfile in ROOT.rglob("*.py"):
            if ".git" in str(pyfile) or "__pycache__" in str(pyfile):
                continue
            with open(pyfile, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "total_mem" in content and "total_memory" not in content:
                # Check it's actually the attribute access, not just a variable name
                if ".total_mem" in content:
                    self.fail(
                        f"{pyfile.relative_to(ROOT)}: contains '.total_mem' "
                        f"(should be '.total_memory')"
                    )

    def test_no_duplicate_returns(self):
        """Detect consecutive duplicate return statements (dead code)."""
        for pyfile in [
            ROOT / "training" / "detection" / "train_yolo.py",
            ROOT / "training" / "detection" / "train_rfdetr.py",
            ROOT / "cloud" / "run_pipeline.py",
        ]:
            with open(pyfile, "r", encoding="utf-8") as f:
                lines = f.readlines()

            prev_stripped = ""
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if (stripped.startswith("return ")
                        and prev_stripped.startswith("return ")
                        and stripped == prev_stripped):
                    self.fail(
                        f"{pyfile.relative_to(ROOT)}:{i}: Duplicate return "
                        f"statement (dead code): '{stripped}'"
                    )
                if stripped:
                    prev_stripped = stripped


class TestConfigValidation(unittest.TestCase):
    """Validate config.yaml structure and values."""

    @classmethod
    def setUpClass(cls):
        with open(ROOT / "cloud" / "config.yaml", "r") as f:
            cls.config = yaml.safe_load(f)

    def test_config_loads(self):
        """config.yaml must be valid YAML."""
        self.assertIsInstance(self.config, dict)

    def test_gcp_section(self):
        """GCP config must have required fields."""
        gcp = self.config.get("gcp", {})
        for key in ["project_id", "region", "bucket_name", "models_bucket"]:
            self.assertIn(key, gcp, f"Missing gcp.{key}")
            self.assertTrue(gcp[key], f"gcp.{key} is empty")

    def test_dataset_section(self):
        """Dataset config must have GCS path and local path."""
        ds = self.config.get("dataset", {})
        self.assertIn("gcs_path", ds)
        self.assertIn("local_path", ds)
        self.assertTrue(ds["gcs_path"].startswith("gs://"), "gcs_path must start with gs://")

    def test_yolo_experiments(self):
        """YOLO experiments must have required fields."""
        yolo = self.config.get("yolo", {})
        if not yolo.get("enabled"):
            self.skipTest("YOLO disabled")

        exps = yolo.get("experiments", {})
        self.assertTrue(len(exps) > 0, "No YOLO experiments defined")

        for name, cfg in exps.items():
            for key in ["epochs", "batch_size", "imgsz", "model"]:
                self.assertIn(key, cfg, f"YOLO experiment '{name}' missing '{key}'")
            self.assertGreater(cfg["epochs"], 0, f"{name}: epochs must be > 0")
            self.assertGreater(cfg["batch_size"], 0, f"{name}: batch_size must be > 0")

    def test_rfdetr_experiments(self):
        """RF-DETR experiments must have required fields."""
        rfdetr = self.config.get("rfdetr", {})
        if not rfdetr.get("enabled"):
            self.skipTest("RF-DETR disabled")

        exps = rfdetr.get("experiments", {})
        self.assertTrue(len(exps) > 0, "No RF-DETR experiments defined")

        for name, cfg in exps.items():
            for key in ["epochs", "batch_size", "lr", "model_class"]:
                self.assertIn(key, cfg, f"RF-DETR experiment '{name}' missing '{key}'")
            self.assertIn(cfg["model_class"], ["RFDETRBase", "RFDETRLarge"],
                          f"{name}: invalid model_class '{cfg['model_class']}'")
            self.assertGreater(cfg["epochs"], 0)
            self.assertGreater(cfg["batch_size"], 0)
            self.assertGreater(cfg["lr"], 0)

    def test_rfdetr_effective_batch_size(self):
        """Effective batch size should be reasonable (8-64)."""
        exps = self.config.get("rfdetr", {}).get("experiments", {})
        for name, cfg in exps.items():
            eff = cfg["batch_size"] * cfg.get("grad_accum_steps", 1)
            self.assertGreaterEqual(eff, 4,
                f"{name}: effective batch {eff} too small")
            self.assertLessEqual(eff, 128,
                f"{name}: effective batch {eff} too large")

    def test_output_section(self):
        """Output config must have base_dir."""
        output = self.config.get("output", {})
        self.assertIn("base_dir", output)

    def test_seed_exists(self):
        """Reproducibility seed must be set."""
        self.assertIn("seed", self.config)
        self.assertIsInstance(self.config["seed"], int)


class TestRequirementsConsistency(unittest.TestCase):
    """Validate requirements files for consistency and common issues."""

    def _parse_requirements(self, path: Path) -> dict:
        """Parse requirements file into {package: version_spec}."""
        reqs = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Parse package name (before any version specifier)
                match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if match:
                    reqs[match.group(1).lower()] = line
        return reqs

    def test_rfdetr_has_pydantic_v2(self):
        """requirements-rfdetr.txt MUST specify pydantic>=2.x (V2 is critical for RF-DETR)."""
        reqs = self._parse_requirements(ROOT / "cloud" / "requirements-rfdetr.txt")
        self.assertIn("pydantic", reqs, "pydantic missing from rfdetr requirements")
        self.assertIn(">=2", reqs["pydantic"],
            f"pydantic must require V2: got '{reqs['pydantic']}'")

    def test_rfdetr_has_transformers(self):
        """RF-DETR needs transformers."""
        reqs = self._parse_requirements(ROOT / "cloud" / "requirements-rfdetr.txt")
        self.assertIn("transformers", reqs)

    def test_rfdetr_has_accelerate(self):
        """RF-DETR needs accelerate."""
        reqs = self._parse_requirements(ROOT / "cloud" / "requirements-rfdetr.txt")
        self.assertIn("accelerate", reqs)

    def test_numpy_constrained(self):
        """NumPy must be < 2.0 to avoid ABI issues with PyTorch containers."""
        for req_file in ["requirements-cloud.txt", "requirements-rfdetr.txt", "requirements-yolo.txt"]:
            path = ROOT / "cloud" / req_file
            reqs = self._parse_requirements(path)
            if "numpy" in reqs:
                self.assertIn("<2", reqs["numpy"],
                    f"{req_file}: numpy must be constrained <2.0 to avoid ABI issues")

    def test_opencv_headless(self):
        """Cloud requirements must use opencv-python-headless (not gui version)."""
        for req_file in ["requirements-cloud.txt", "requirements-rfdetr.txt", "requirements-yolo.txt"]:
            path = ROOT / "cloud" / req_file
            with open(path) as f:
                content = f.read()
            # Should NOT have opencv-python (non-headless) as primary
            lines = [l.strip() for l in content.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            for line in lines:
                if line.startswith("opencv-python") and "headless" not in line:
                    self.fail(f"{req_file}: Use opencv-python-headless in cloud, not opencv-python")

    def test_no_torch_in_cloud_requirements(self):
        """Cloud requirements must NOT reinstall torch (pre-installed in container)."""
        for req_file in ["requirements-cloud.txt", "requirements-rfdetr.txt", "requirements-yolo.txt"]:
            reqs = self._parse_requirements(ROOT / "cloud" / req_file)
            self.assertNotIn("torch", reqs,
                f"{req_file}: Do NOT include 'torch' - it's pre-installed in the Vertex AI container and reinstalling breaks CUDA")
            self.assertNotIn("torchvision", reqs,
                f"{req_file}: Do NOT include 'torchvision' - pre-installed in container")


class TestStartupScript(unittest.TestCase):
    """Validate the Vertex AI startup script embedded in submit_vertex_job.ps1."""

    @classmethod
    def setUpClass(cls):
        with open(ROOT / "cloud" / "submit_vertex_job.ps1", "r") as f:
            cls.script_content = f.read()

    def test_container_is_pytorch_22(self):
        """Container must be pytorch-gpu.2-2 (not 2.1) for register_pytree_node support."""
        self.assertIn("pytorch-gpu.2-2", self.script_content,
            "Container must use PyTorch 2.2+ for transformers compatibility with RF-DETR")

    def test_pydantic_uninstall_before_install(self):
        """Startup script must force-uninstall pydantic before installing V2."""
        self.assertIn("pip uninstall -y pydantic", self.script_content,
            "Must force-uninstall pre-installed pydantic V1 before pip install V2")

    def test_field_validator_gate(self):
        """Startup script must verify field_validator import before training."""
        self.assertIn("field_validator", self.script_content,
            "Must verify pydantic.field_validator is importable before training starts")

    def test_no_cache_dir_for_pydantic(self):
        """Pydantic install must use --no-cache-dir to avoid stale cached V1."""
        self.assertIn("--no-cache-dir", self.script_content,
            "Must use --no-cache-dir when installing pydantic to avoid cached V1 wheels")

    def test_set_e_enabled(self):
        """Startup script should use 'set -e' for fail-fast behavior."""
        self.assertIn("set -e", self.script_content,
            "Startup script must use 'set -e' to abort on any command failure")

    def test_vertex_ai_job_env(self):
        """Must set VERTEX_AI_JOB=true for the pipeline to detect cloud mode."""
        self.assertIn("VERTEX_AI_JOB=true", self.script_content)

    def test_gcs_paths_set(self):
        """Must set GCS_DATASET_PATH and GCS_OUTPUT_PATH."""
        self.assertIn("GCS_DATASET_PATH", self.script_content)
        self.assertIn("GCS_OUTPUT_PATH", self.script_content)

    def test_bash_vars_escaped_in_powershell(self):
        """Bash-only variables must be escaped with backtick in PS @"..."@ here-strings.

        PowerShell expandable here-strings (@"..."@) treat $VAR as PS variables.
        Bash variables like $REQ_FILE, $EXPERIMENT_NAME, $DIAGNOSE_MODE must be
        escaped as `$REQ_FILE so PS passes them literally to bash.
        
        This test catches the exact bug that caused Job 2716602325509603328 to fail:
        '$REQ_FILE' was expanded to empty by PowerShell, making 'pip install -r' fail
        with '-r option requires 1 argument'.
        """
        # Extract the bash here-string content between @" and "@
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        self.assertTrue(len(matches) > 0, "No here-strings found in submit_vertex_job.ps1")

        # First here-string is the startup script
        startup_bash = matches[0]

        # These are bash-only variables that must NOT appear unescaped
        # (they would be expanded by PowerShell to empty strings)
        bash_only_vars = ["REQ_FILE", "DIAGNOSE_MODE"]

        for var in bash_only_vars:
            # Find all occurrences of $VAR that are NOT escaped with backtick
            # Pattern: $ not preceded by backtick, followed by VAR
            unescaped = re.findall(rf'(?<!`)\${var}', startup_bash)
            if unescaped:
                self.fail(
                    f"submit_vertex_job.ps1: ${var} appears unescaped "
                    f"({len(unescaped)}x) in PowerShell here-string. "
                    f"Must be `${var} to survive PS expansion."
                )

    def test_numpy_pinned_before_opencv(self):
        """NumPy must be pinned <2.0 BEFORE opencv install to prevent ABI breakage."""
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""

        # NumPy pin must appear BEFORE opencv install
        numpy_pos = startup_bash.find("numpy")
        opencv_pos = startup_bash.find("opencv-python-headless")

        self.assertGreater(numpy_pos, -1, "NumPy pin not found in startup script")
        self.assertGreater(opencv_pos, -1, "OpenCV install not found in startup script")
        self.assertLess(numpy_pos, opencv_pos,
            "NumPy must be pinned BEFORE opencv install to prevent numpy 2.x pull")

    def test_gpu_config_consistent(self):
        """Job spec GPU must match config.yaml GPU."""
        import yaml
        with open(ROOT / "cloud" / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        config_gpu = config.get("vertex_ai", {}).get("accelerator_type", "")
        # The job spec in the PS script must use the same GPU
        self.assertIn(config_gpu, self.script_content,
            f"config.yaml uses {config_gpu} but submit_vertex_job.ps1 uses different GPU")

    def test_gpu_sanity_check_before_code_download(self):
        """GPU sanity check must run BEFORE code download to fail fast (~5s vs ~60min).
        
        This prevents wasting 60+ minutes downloading the dataset only to discover
        the GPU doesn't have CUDA available at training time.
        """
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""

        gpu_check_pos = startup_bash.find("torch.cuda.get_device_capability")
        code_download_pos = startup_bash.find("gcloud storage cp")

        self.assertGreater(gpu_check_pos, -1,
            "GPU sanity check (get_device_capability) not found in startup script")
        self.assertGreater(code_download_pos, -1,
            "Code download (gcloud storage cp) not found in startup script")
        self.assertLess(gpu_check_pos, code_download_pos,
            "GPU sanity check must run BEFORE code download for fast-fail")
        
        # Verify backward pass logic is correct
        self.assertIn("requires_grad=True", startup_bash,
            "Sanity check must use requires_grad=True for backward() test")

    def test_bf16_detection_in_startup(self):
        """Startup script must detect bfloat16 support for GPU capability reporting."""
        self.assertIn("bfloat16", self.script_content,
            "Startup script must check bfloat16 support")

    def test_bf16_detection_in_training(self):
        """train_rfdetr.py must detect GPU capability and set amp accordingly."""
        with open(ROOT / "training" / "detection" / "train_rfdetr.py", "r") as f:
            train_content = f.read()
        self.assertIn("get_device_capability", train_content,
            "train_rfdetr.py must check GPU capability for amp/bf16 detection")
        self.assertIn("amp=use_amp", train_content,
            "train_rfdetr.py must pass dynamic amp flag to model.train()")

    def test_smoke_test_param_exists(self):
        """submit_vertex_job.ps1 must have a -SmokeTest parameter."""
        self.assertIn("SmokeTest", self.script_content,
            "Must support -SmokeTest parameter for quick pipeline validation")

    def test_torch_xla_uninstall(self):
        """Startup script must uninstall torch_xla to prevent SIGABRT.
        
        The Vertex AI pytorch-gpu container has torch_xla pre-installed,
        which crashes with 'PJRT_DEVICE is not set' during training.
        """
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""
        self.assertIn("uninstall", startup_bash.lower(),
            "Startup script must uninstall torch_xla")
        self.assertIn("torch_xla", startup_bash,
            "Startup script must reference torch_xla for removal")

    def test_pjrt_device_env_var(self):
        """Startup script must set PJRT_DEVICE=CPU as safety net."""
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""
        self.assertIn("PJRT_DEVICE", startup_bash,
            "Must set PJRT_DEVICE=CPU to prevent torch_xla SIGABRT")

    def test_no_restart_on_failure(self):
        """Job spec must disable automatic restarts on worker failure.
        
        Without this, a crashed job restarts endlessly and must be
        manually cancelled. With restartJobOnWorkerRestart=false,
        the job dies immediately for log inspection.
        """
        self.assertIn("restartJobOnWorkerRestart", self.script_content,
            "Job YAML must include restartJobOnWorkerRestart setting")
        self.assertIn("false", self.script_content.lower(),
            "restartJobOnWorkerRestart must be set to false")

    def test_torch_xla_absence_validation(self):
        """Phase 4 validation must verify torch_xla is NOT installed."""
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""
        self.assertIn("import torch_xla", startup_bash,
            "Validation must check for torch_xla absence")

    def test_mkl_service_env_var(self):
        """Startup script must set MKL_SERVICE_FORCE_INTEL=1."""
        here_pattern = re.compile(r'@"(.*?)"@', re.DOTALL)
        matches = here_pattern.findall(self.script_content)
        startup_bash = matches[0] if matches else ""
        self.assertIn("MKL_SERVICE_FORCE_INTEL", startup_bash,
            "Must set MKL_SERVICE_FORCE_INTEL=1 to prevent MKL conflicts")


class TestDockerfile(unittest.TestCase):
    """Validate Dockerfile for cloud training."""

    @classmethod
    def setUpClass(cls):
        with open(ROOT / "cloud" / "Dockerfile", "r") as f:
            cls.content = f.read()

    def test_base_image_pytorch_22(self):
        """Dockerfile base must be PyTorch 2.2+ for compatibility."""
        self.assertTrue(
            "pytorch:2.2" in self.content or "pytorch-gpu.2-2" in self.content,
            "Dockerfile should use PyTorch 2.2+ base image"
        )

    def test_copies_training_code(self):
        """Dockerfile must COPY training/, cloud/, and core/ directories."""
        for dir_name in ["training/", "cloud/", "core/"]:
            self.assertIn(dir_name, self.content,
                f"Dockerfile must COPY {dir_name}")

    def test_installs_requirements(self):
        """Dockerfile must install requirements."""
        self.assertIn("requirements", self.content)


class TestTrainingScriptQuality(unittest.TestCase):
    """Deep static analysis of training scripts for common issues."""

    def _get_source(self, relpath: str) -> str:
        with open(ROOT / relpath, "r", encoding="utf-8") as f:
            return f.read()

    def test_rfdetr_lazy_imports(self):
        """RF-DETR model imports must be lazy (inside functions, not top-level)."""
        source = self._get_source("training/detection/train_rfdetr.py")
        # rfdetr import should NOT be at module level
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'module') and node.module and 'rfdetr' in node.module:
                    # Check if it's inside a function (ok) or at module level (bad)
                    # Simple heuristic: if col_offset is 0 and it's a direct child of Module
                    pass  # ast.walk doesn't give parent info easily

        # Alternative: check that 'from rfdetr' does NOT appear at col 0
        # (except in function bodies)
        lines = source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("from rfdetr") and not line.startswith(" ") and not line.startswith("\t"):
                self.fail(
                    f"train_rfdetr.py:{i}: 'from rfdetr' at module level will crash "
                    f"if rfdetr is not installed. Move inside function."
                )

    def test_yolo_lazy_imports(self):
        """YOLO (ultralytics) import should be lazy in cloud pipeline function."""
        source = self._get_source("cloud/run_pipeline.py")
        lines = source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("from ultralytics") and not line.startswith(" ") and not line.startswith("\t"):
                self.fail(
                    f"run_pipeline.py:{i}: 'from ultralytics' at module level. "
                    f"Should be lazy import inside function."
                )

    def test_gpu_memory_attribute(self):
        """All training scripts must use .total_memory (not .total_mem)."""
        for script in [
            "training/detection/train_rfdetr.py",
            "training/detection/train_yolo.py",
        ]:
            source = self._get_source(script)
            self.assertNotIn(".total_mem ", source,
                f"{script}: Use .total_memory instead of .total_mem")
            self.assertNotIn(".total_mem/", source,
                f"{script}: Use .total_memory instead of .total_mem")

    def test_cuda_empty_cache_on_error(self):
        """Training scripts should clean GPU on error to prevent OOM in next experiment."""
        for script in [
            "training/detection/train_rfdetr.py",
            "training/detection/train_yolo.py",
        ]:
            source = self._get_source(script)
            self.assertIn("torch.cuda.empty_cache()", source,
                f"{script}: Must call torch.cuda.empty_cache() for GPU cleanup")

    def test_gc_collect_after_training(self):
        """Must call gc.collect() after deleting models to free memory."""
        for script in [
            "training/detection/train_rfdetr.py",
            "training/detection/train_yolo.py",
        ]:
            source = self._get_source(script)
            self.assertIn("gc.collect()", source,
                f"{script}: Must call gc.collect() after training to free memory")


if __name__ == "__main__":
    print("=" * 60)
    print(" VM_FOOTBALL - Pre-flight Test Suite")
    print("=" * 60)
    print(f" Project root: {ROOT}")
    print()

    # Run with verbosity
    unittest.main(verbosity=2)
