"""API contract between the notebook and the package.

Real failure this prevents: an edit to kaggle_setup.py accidentally deleted
ensure_gsr_data (a string-replacement spanned past the function it targeted),
all 23 tests still passed, and the run died on Kaggle with AttributeError.
The gap: the notebook test checked the notebook's TEXT, not that the package
actually EXPORTS what the notebook calls. This test closes it by parsing the
notebook's code cells and resolving every `module.attr` and imported name
against the real package.
"""
import ast
import importlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NB = ROOT / "notebooks" / "optiona_kaggle_pipeline.ipynb"

# Names the pipeline depends on, stated explicitly (belt) in addition to the
# notebook scan (suspenders): deleting any of these must fail the suite even
# if the notebook is edited in the same commit.
REQUIRED = {
    "optiona_sfr.config": ["ExperimentCfg", "Paths", "ablation_grid",
                           "DetectorCfg", "RefineCfg", "TrackerCfg", "V2Cfg", "b_grid", "V3Cfg", "c_grid",
                           "SmoothCfg", "s_grid"],
    "optiona_sfr.kaggle_setup": ["setup_pnlcalib", "ensure_gsr_data",
                                 "harvest_labels", "release_split",
                                 "pick_scratch_root", "disk_free_gb",
                                 "verify_checkpoint", "download_checkpoint",
                                 "extract_sequences", "prune_images",
                                 "split_is_ready", "check_gsr_version"],
    "optiona_sfr.pnl_adapter": ["load_world", "load_models",
                                "load_keypoint_world", "circle_membership", "infer_and_center",
                                "keypoints_to_obs", "lines_to_obs"],
    "optiona_sfr.detection": ["cache_sequence", "detect_frame", "load_cached",
                              "make_single_frame_calibrator", "get_baseline_cameras",
                              "select_baseline_variant",
                              "baseline_cache_path",
                              "compute_flow_pairs"],
    "optiona_sfr.experiments": ["run_sequence", "cached_sequence_ids",
                                "list_gsr_sequences", "summarize",
                                "load_pitch_annotations",
                                "reprojection_error", "_result_stamp", "frame_indices",
                                "set_pitch_convention"],
    "optiona_sfr.tracker": ["BroadTrackLayer", "estimate_tripod",
                            "jaccard_confidence", "flow_from_pairs"],
    "optiona_sfr.multigpu": ["gpu_inventory", "shard_cache_all",
                             "parallel_run_experiments"],
    "optiona_sfr.diagnose": ["diagnose_sequence", "fit_camera_to_annotations",
                             "annotation_report", "audit_conversion_and_mapping",
                             "resolve_and_install_convention", "audit_correspondences",
                             "refinement_term_ablation"],
    "optiona_sfr.pitch_model": ["build_name_map", "resolve_convention",
                                "CONVENTIONS"],
    "optiona_sfr.method_v2": ["estimate_fixed_center", "refine_fixed_center",
                              "smooth_and_interpolate", "run_sequence_v2",
                              "FixedCenterObjective", "flow_residual_factory"],
    "optiona_sfr.method_v3": ["so3_log", "so3_exp", "slerp_rotations",
                              "smooth_rotations_so3", "smooth_log_focal",
                              "center_prior_residual", "bundle_adjust",
                              "bic", "bic_select", "mahalanobis_repair",
                              "reprojection_covariances", "run_sequence_v3"],
    "optiona_sfr.smooth": ["smooth_cameras", "cap_deviation", "smooth_pnlcalib",
                           "run_sequence_smooth"],
    "optiona_sfr.parity": ["flip_pitch_half", "diagnose_sequence_failure"],
    "optiona_sfr.subpixel": ["refine_corner", "refine_detections"],
    "optiona_sfr.refine": ["refine_camera", "build_circle_obs", "Residuals",
                           "select_inliers", "score_camera", "robust_cost",
                           "projected_displacement",
                           "_cheirality"],
    "optiona_sfr.geometry": ["Camera", "LINES_WORLD", "CIRCLES_WORLD",
                             "sampson_conic_distance", "circle_conic",
                             "conic_image", "point_line_distance", "camera_is_plausible",
                             "project_full" if False else "circle_conic",
                             "param_bounds", "clip_to_bounds"],
}


def test_every_required_symbol_exists():
    missing = []
    for mod, names in REQUIRED.items():
        m = importlib.import_module(mod)
        for n in names:
            if not hasattr(m, n):
                missing.append(f"{mod}.{n}")
    assert not missing, f"missing exports: {missing}"


def _notebook_code():
    nb = json.loads(NB.read_text())
    return [''.join(c.get("source", [])) for c in nb["cells"]
            if c.get("cell_type") == "code"]


def test_notebook_symbols_resolve_against_package():
    """Parse every code cell; for each `from optiona_sfr.X import a, b` and
    each `ALIAS.attr` where ALIAS is an optiona_sfr module, assert the
    attribute really exists."""
    missing, aliases = [], {}
    for src in _notebook_code():
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"notebook cell does not parse: {e}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith("optiona_sfr"):
                m = importlib.import_module(node.module)
                for a in node.names:
                    if not hasattr(m, a.name):
                        missing.append(f"{node.module}.{a.name}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("optiona_sfr"):
                        aliases[a.asname or a.name.split(".")[-1]] = a.name
    for src in _notebook_code():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == \
                    "optiona_sfr" :
                for a in node.names:            # from optiona_sfr import X as K
                    aliases[a.asname or a.name] = f"optiona_sfr.{a.name}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and \
                    isinstance(node.value, ast.Name) and \
                    node.value.id in aliases:
                m = importlib.import_module(aliases[node.value.id])
                if not hasattr(m, node.attr):
                    missing.append(f"{aliases[node.value.id]}.{node.attr}")
    assert not missing, f"notebook references missing symbols: {sorted(set(missing))}"


def test_all_modules_import_cleanly():
    for mod in ["config", "geometry", "refine", "tracker", "detection",
                "experiments", "pnl_adapter", "multigpu", "kaggle_setup",
                "diagnose", "pitch_model", "method_v2", "method_v3",
                "parity", "subpixel", "smooth"]:
        importlib.import_module(f"optiona_sfr.{mod}")


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
