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


def test_new_margin_configs_are_documented():
    readme = (ROOT / "README.md").read_text()

    for config_name in [
        "magface",
        "circle",
        "partialfc_arcface",
        "distributed_partialfc_arcface",
        "sphereface",
    ]:
        config_path = ROOT / "configs" / "margin" / f"{config_name}.yaml"
        assert config_path.is_file()
        assert f"configs/margin/{config_name}.yaml" in readme


def test_new_loss_configs_exist():
    for config_name in [
        "circle",
        "uniface_uce",
        "topofr",
        "topofr_sde",
        "transface_ehsm",
        "multi_similarity",
        "batch_hard_triplet",
        "supcon",
        "ntxent",
        "proxy_anchor",
        "proxy_nca",
        "sface",
        "boundaryface",
        "unitsface",
        "focal_loss",
        "soft_cross_entropy",
    ]:
        assert (ROOT / "configs" / "loss" / f"{config_name}.yaml").is_file()


def test_new_dataloader_configs_exist():
    for config_name in ["pk", "class_balanced", "hierarchical_pk", "hard_negative"]:
        assert (ROOT / "configs" / "dataloader" / f"{config_name}.yaml").is_file()


def test_face_evaluator_configs_exist():
    for config_name in ["face_verification", "ijb_template"]:
        config_path = (
            ROOT / "configs" / "evaluation" / "evaluator" / f"{config_name}.yaml"
        )
        assert config_path.is_file()


def test_advanced_optimizer_configs_exist():
    for config_name in ["lamb", "muon", "schedule_free_adamw"]:
        assert (ROOT / "configs" / "optimizer" / f"{config_name}.yaml").is_file()


def test_strong_augmentation_configs_exist():
    for config_name in ["randaugment", "trivialaugment", "augmix"]:
        assert (ROOT / "configs" / "transform" / f"{config_name}.yaml").is_file()


def test_faiss_index_configs_exist():
    for config_name in ["faiss_ivf", "faiss_hnsw"]:
        assert (
            ROOT / "configs" / "evaluation" / "knn" / f"{config_name}.yaml"
        ).is_file()


def test_new_backbone_configs_exist():
    for config_name in [
        "dinov2_vit_b14",
        "dinov2_vit_l14",
        "dinov3_vit_s16",
        "dinov3_vit_b16",
        "dinov3_vit_l16",
        "dinov3_convnext_tiny",
        "siglip_vit_b16",
        "siglip2_vit_so400m_14",
        "aimv2_vit_l14",
        "eva02_vit_b14",
        "hiera_base_plus",
        "convnextv2_base",
        "fastvit_sa24",
        "edgeface_base",
        "edgeface_s_gamma_05",
        "edgeface_xs_gamma_06",
        "edgeface_xxs",
        "mobilefacenet",
        "am_radio_v2_5_b",
        "kprpe_vit_base_patch16_224",
        "iresnet50",
        "iresnet100",
        "iresnet200",
        "openclip_mobileclip_s0",
        "openclip_mobileclip_s1",
        "openclip_mobileclip_s2",
    ]:
        config_path = ROOT / "configs" / "backbone" / f"{config_name}.yaml"
        assert config_path.is_file()


def test_backbone_configs_declare_family():
    offenders = []
    for config_path in (ROOT / "configs" / "backbone").glob("*.yaml"):
        if "family:" not in config_path.read_text():
            offenders.append(config_path.name)

    assert offenders == []


def test_manual_hydra_tools_do_not_reference_missing_config_yaml():
    offenders = []
    for path in (ROOT / "tools").rglob("test*.py"):
        text = path.read_text()
        if "config_name='config'" in text or 'config_name="config"' in text:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_nearest_search_uses_current_knn_helpers():
    nearest_search = (ROOT / "tools" / "nearest_search.py").read_text()

    assert "from src.evaluator.knn.base import" in nearest_search
    assert "from src.utils import cosine_similarity_chunks" not in nearest_search


def test_mobile_embedding_compare_converts_cv2_images_to_rgb():
    compare_tool = (
        ROOT / "tools" / "knowledge_distillation" / "compare_embeddings_mobile.py"
    ).read_text()

    assert "cv2.COLOR_BGR2RGB" in compare_tool


def test_lamb_optimizer_does_not_sync_trust_ratio_to_cpu():
    lamb = (ROOT / "src" / "optimizer" / "lamb.py").read_text()

    assert "trust_ratio.item()" not in lamb


def test_runtime_code_uses_current_torch_amp_api():
    offenders = []
    for path in [ROOT / "src", ROOT / "tools"]:
        for python_file in path.rglob("*.py"):
            text = python_file.read_text()
            if "torch.cuda.amp" in text or "from torch.cuda import amp" in text:
                offenders.append(python_file.relative_to(ROOT).as_posix())

    assert offenders == []


def test_distillation_teacher_config_uses_placeholders_not_local_paths():
    teacher = (
        ROOT
        / "configs"
        / "distillation"
        / "teacher"
        / "arcface_openclip_vit_b16_retail_full.yaml"
    ).read_text()

    assert "work_dirs/train" not in teacher
    assert "???" in teacher


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
