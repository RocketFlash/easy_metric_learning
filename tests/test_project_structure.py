from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pytest_collects_dedicated_tests_not_manual_tools():
    pytest_ini = (ROOT / "pytest.ini").read_text()

    assert "testpaths = tests" in pytest_ini


def test_repo_has_editable_install_metadata():
    assert (ROOT / "pyproject.toml").is_file()


def test_pyproject_declares_test_extra():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert "[project.optional-dependencies]" in pyproject
    assert "test = [" in pyproject
    assert "pytest" in pyproject


def test_cpu_pytest_workflow_exists():
    workflow = ROOT / ".github" / "workflows" / "test.yml"

    assert workflow.is_file()
    assert "pytest -q" in workflow.read_text()


def test_requirements_include_test_and_runtime_basics_without_gpu_onnxruntime():
    requirements = (ROOT / "requirements.txt").read_text()

    assert "torch\n" in requirements
    assert "torchvision\n" in requirements
    assert "pytest\n" in requirements
    assert "kaggle\n" in requirements
    assert "onnxruntime-gpu" not in requirements


def test_environment_file_does_not_pin_author_prefix():
    environment = (ROOT / "environment.yml").read_text()

    assert "prefix:" not in environment


def test_worker_init_does_not_use_numpy_internal_rng_state():
    data_utils = (ROOT / "src" / "data" / "utils.py").read_text()

    assert "np.random.get_state" not in data_utils
    assert "torch.initial_seed" in data_utils


def test_documented_training_config_exists():
    readme = (ROOT / "README.md").read_text()

    assert "config_train.yaml" in readme
    assert (ROOT / "configs" / "config_train.yaml").is_file()


def test_manual_hydra_tools_do_not_reference_missing_config_yaml():
    offenders = []
    for path in (ROOT / "tools").rglob("test*.py"):
        text = path.read_text()
        if "config_name='config'" in text or 'config_name="config"' in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_data_scripts_do_not_use_shell_execution():
    offenders = []
    for path in [
        ROOT / "data" / "datasets" / "hnm.py",
        ROOT / "data" / "datasets" / "shopee.py",
        ROOT / "data" / "zip_dataset.py",
    ]:
        if "os.system" in path.read_text():
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_training_config_does_not_hardcode_internal_mlflow_uri():
    config = (ROOT / "configs" / "config_train.yaml").read_text()

    assert "r3dev-mlflow.rebotics.net" not in config


def test_mlflow_is_skipped_when_tracking_uri_is_unset():
    tracker_factory = (ROOT / "src" / "experiment_tracker" / "__init__.py").read_text()

    assert "mlflow tracking uri is not configured" in tracker_factory
    assert "skipping mlflow" in tracker_factory
