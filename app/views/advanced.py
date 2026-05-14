"""
Page: Advanced Configuration

Lets the user re-run the pipeline on the same dataset with different settings.
The basic Upload page only exposes three knobs (imputation, dim. reduction,
clustering); this page exposes every parameter the AEDAPipeline accepts.

Two modes:
- Default: high-level method choices only (auto / pearson / kmeans / ...).
  Recommended for non-expert users.
- Expert overrides: an opt-in toggle that unlocks fine-grained kwargs
  (k of K-Means, eps of DBSCAN, n_components of PCA, etc.) for the
  scientist who wants to dig into the ML internals.

The page reads the previously used settings from `st.session_state.run_context`
and pre-fills its controls so the user always sees a known starting point.
The "Re-run" button executes the pipeline with the current data and writes
new results back into `st.session_state.results`.
"""

import time
import streamlit as st


def render():
    from app.components.page_header import page_header

    page_header(
        title="Advanced Configuration",
        description=(
            "Re-run the analysis on the currently loaded dataset with custom "
            "parameters. Useful for sensitivity analysis and for the scientific "
            "tutor to validate alternative methodological choices."
        ),
        icon="⚙️",
    )

    ctx = st.session_state.get("run_context")
    if ctx is None or st.session_state.get("results") is None:
        st.info(
            "Upload a dataset from the Upload page first. "
            "Once an initial run has been completed, this page will let you "
            "re-run the same analysis with different settings."
        )
        return

    settings = dict(ctx["settings"])  # copy: do not mutate session state
    raw_df = st.session_state.results.raw_data

    # ---- Pre-flight info ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", st.session_state.get("filename", "—"))
    col2.metric("Samples", raw_df.shape[0] if raw_df is not None else 0)
    col3.metric("Variables", raw_df.shape[1] if raw_df is not None else 0)

    st.divider()

    # ---- Standard controls ----
    new_settings = _render_standard_controls(settings)

    # ---- Environmental interpretation block ----
    st.divider()
    new_settings.update(_render_interpretation_controls(settings, raw_df))

    # ---- Expert overrides ----
    st.divider()
    expert = st.toggle(
        "Expert overrides (fine-grained ML parameters)",
        value=False,
        help=(
            "Unlock manual control over algorithm-specific parameters. "
            "Leave off to keep the system in fully automatic mode."
        ),
    )
    if expert:
        new_settings.update(_render_expert_controls(settings))
    else:
        # Keep whatever was there before (likely empty dicts)
        new_settings["dim_kwargs"] = settings.get("dim_kwargs", {})
        new_settings["clustering_kwargs"] = settings.get("clustering_kwargs", {})
        new_settings["anomaly_kwargs"] = settings.get("anomaly_kwargs", {})

    st.divider()

    # ---- Diff vs. last run ----
    diff = _settings_diff(settings, new_settings)
    if diff:
        with st.expander(f"Changes vs. last run ({len(diff)} parameter(s))", expanded=True):
            _show_diff_table(diff)
    else:
        st.caption("No changes vs. the last run.")

    # ---- Run button ----
    if st.button(
        "Re-run pipeline with these settings",
        type="primary",
        use_container_width=True,
        disabled=not diff,
    ):
        _rerun_pipeline(ctx, new_settings)


# ============================================================
# Standard controls
# ============================================================
def _render_standard_controls(settings: dict) -> dict:
    """Render the high-level method choices that AEDAPipeline accepts."""
    out = {}

    st.subheader("Pipeline configuration")

    # Preprocessing row
    st.markdown("**Preprocessing**")
    col1, col2, col3 = st.columns(3)
    with col1:
        out["impute_strategy"] = st.selectbox(
            "Missing value strategy",
            options=["auto", "median", "mean", "knn", "drop_rows", "drop_cols"],
            index=_safe_index(
                ["auto", "median", "mean", "knn", "drop_rows", "drop_cols"],
                settings.get("impute_strategy", "auto"),
            ),
            help="How to fill in or remove missing values.",
        )
    with col2:
        out["scale_method"] = st.selectbox(
            "Scaling method",
            options=["auto", "standard", "minmax", "robust"],
            index=_safe_index(
                ["auto", "standard", "minmax", "robust"],
                settings.get("scale_method", "auto"),
            ),
            help="Robust scaling resists outliers; standard is the typical default.",
        )
    with col3:
        clr_choice = st.selectbox(
            "CLR transform (compositional)",
            options=["auto", "off", "on"],
            index={None: 0, "auto": 0, False: 1, True: 2}.get(
                settings.get("apply_clr"), 0
            ),
            help=(
                "Apply Centered Log-Ratio transform for compositional data "
                "(XRF, granulometry). 'auto' lets the brain decide."
            ),
        )
        out["apply_clr"] = {"auto": "auto", "off": False, "on": True}[clr_choice]

    # Engine methods row
    st.markdown("**Analysis methods**")
    col1, col2, col3 = st.columns(3)
    with col1:
        out["dim_method"] = st.selectbox(
            "Dimensionality reduction",
            options=["auto", "pca", "tsne", "umap"],
            index=_safe_index(
                ["auto", "pca", "tsne", "umap"], settings.get("dim_method", "auto")
            ),
            help="PCA is the standard choice for environmental EDA.",
        )
    with col2:
        out["clustering_method"] = st.selectbox(
            "Clustering",
            options=["auto", "kmeans", "dbscan", "hierarchical"],
            index=_safe_index(
                ["auto", "kmeans", "dbscan", "hierarchical"],
                settings.get("clustering_method", "auto"),
            ),
            help="'auto' picks the best between K-Means and DBSCAN by silhouette.",
        )
    with col3:
        out["anomaly_method"] = st.selectbox(
            "Anomaly detection",
            options=["auto", "isolation_forest", "lof", "zscore", "iqr"],
            index=_safe_index(
                ["auto", "isolation_forest", "lof", "zscore", "iqr"],
                settings.get("anomaly_method", "auto"),
            ),
        )

    col1, col2 = st.columns(2)
    with col1:
        out["correlation_method"] = st.selectbox(
            "Correlation method",
            options=["compare", "pearson", "spearman"],
            index=_safe_index(
                ["compare", "pearson", "spearman"],
                settings.get("correlation_method", "compare"),
            ),
            help="'compare' computes both Pearson and Spearman.",
        )
    with col2:
        out["contamination"] = st.slider(
            "Anomaly contamination rate",
            min_value=0.01,
            max_value=0.30,
            value=float(settings.get("contamination", 0.05)),
            step=0.01,
            help="Expected fraction of anomalous samples in the dataset.",
        )

    return out


# ============================================================
# Environmental interpretation controls
# ============================================================
def _render_interpretation_controls(settings: dict, raw_df) -> dict:
    """Render the controls for the EF / TEL/PEL interpretation layer."""
    out = {}

    st.subheader("Environmental interpretation")
    st.caption(
        "Configure how enrichment factors and toxicological classifications "
        "are computed. The reference element should be conservative "
        "(typically Al or Fe) and present in the dataset."
    )

    out["run_interpretation"] = st.checkbox(
        "Run environmental interpretation (EF, TEL/PEL, Birch)",
        value=bool(settings.get("run_interpretation", True)),
    )

    if not out["run_interpretation"]:
        # Carry the rest forward without showing controls.
        out["reference_element"] = settings.get("reference_element", "Al")
        out["baseline_strategy"] = settings.get("baseline_strategy", "deepest")
        out["custom_baseline"] = settings.get("custom_baseline")
        return out

    # Reference element: build options from the dataset's numeric columns,
    # prioritizing common conservative references.
    available = list(raw_df.select_dtypes(include="number").columns) if raw_df is not None else []
    preferred = [c for c in ("Al", "Fe", "Sc", "Li", "Ti") if c in available]
    others = [c for c in available if c not in preferred]
    options = preferred + others or [settings.get("reference_element", "Al")]

    col1, col2 = st.columns(2)
    with col1:
        out["reference_element"] = st.selectbox(
            "Reference element",
            options=options,
            index=_safe_index(options, settings.get("reference_element", "Al")),
            help=(
                "Conservative element used as a normalizer in the EF formula. "
                "Al and Fe are the most common choices for sediment studies."
            ),
        )
    with col2:
        out["baseline_strategy"] = st.selectbox(
            "Baseline strategy",
            options=["deepest", "global_min_depth", "user"],
            index=_safe_index(
                ["deepest", "global_min_depth", "user"],
                settings.get("baseline_strategy", "deepest"),
            ),
            help=(
                "deepest: per-site deepest sample (recommended when sites are present). "
                "global_min_depth: single deepest sample for the whole dataset. "
                "user: provide your own baseline values."
            ),
        )

    if out["baseline_strategy"] == "user":
        out["custom_baseline"] = _render_custom_baseline_editor(
            settings.get("custom_baseline")
        )
    else:
        out["custom_baseline"] = None

    return out


def _render_custom_baseline_editor(current):
    """Allow the user to edit a custom baseline as a table using st.data_editor.

    Two accepted formats (also accepted by `_validate_user_baseline`):
    - Flat dict: {"Al": 8.2, "Pb": 14.0, ...}
    - Per-site:  {"site1": {"Al": ..., "Pb": ...}, "site2": {...}}
    """
    import pandas as pd

    st.markdown("**Custom baseline**")
    st.caption(
        "Edit baseline values as a table. The reference element and all "
        "metals to be analyzed must be present."
    )

    # Convert current dict to DataFrame for editing
    if current is None:
        # Default: flat dict with a few elements
        baseline_dict = {"Al": 0.0, "Pb": 0.0, "Zn": 0.0}
    elif isinstance(current, dict) and all(isinstance(v, dict) for v in current.values()):
        # Per-site format; flatten to single row for simplicity
        # (in production, might want a multi-row editor)
        baseline_dict = next(iter(current.values()))
    else:
        # Flat format
        baseline_dict = current or {}

    df_edit = pd.DataFrame(list(baseline_dict.items()), columns=["Element", "Concentration"])

    edited_df = st.data_editor(
        df_edit,
        num_rows="dynamic",
        use_container_width=True,
        height=200,
    )

    # Convert back to dict
    if edited_df is not None and len(edited_df) > 0:
        try:
            result = dict(zip(edited_df["Element"], edited_df["Concentration"]))
            st.success("Baseline parsed correctly.")
            return result
        except (ValueError, KeyError) as e:
            st.error(f"Error parsing baseline: {e}")
            return None
    return None


# ============================================================
# Expert overrides
# ============================================================
def _render_expert_controls(settings: dict) -> dict:
    """Fine-grained kwargs for each engine method.

    Values are forwarded to `reduce()`, `cluster()`, `detect_anomalies()`
    via the `dim_kwargs`, `clustering_kwargs`, `anomaly_kwargs` parameters
    of AEDAPipeline. Each engine function uses **kwargs and ignores values
    it does not recognize, so it is safe to pass parameters that may not
    apply to the currently selected method.
    """
    st.subheader("Expert overrides")
    st.caption(
        "These parameters override the automatic choices of each ML method. "
        "Leave them at their default to keep the system in auto mode."
    )

    out = {
        "dim_kwargs": dict(settings.get("dim_kwargs") or {}),
        "clustering_kwargs": dict(settings.get("clustering_kwargs") or {}),
        "anomaly_kwargs": dict(settings.get("anomaly_kwargs") or {}),
    }

    # ---- PCA / dim reduction ----
    with st.expander("Dimensionality reduction (PCA / t-SNE / UMAP)", expanded=False):
        n_comp = st.number_input(
            "Number of components (0 = automatic)",
            min_value=0,
            max_value=50,
            value=int(out["dim_kwargs"].get("n_components") or 0),
            step=1,
        )
        if n_comp > 0:
            out["dim_kwargs"]["n_components"] = int(n_comp)
        else:
            out["dim_kwargs"].pop("n_components", None)

    # ---- Clustering ----
    with st.expander("Clustering parameters", expanded=False):
        st.caption("K-Means")
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.number_input(
                "n_clusters (0 = automatic)",
                min_value=0,
                max_value=20,
                value=int(out["clustering_kwargs"].get("n_clusters") or 0),
                step=1,
                help="Number of clusters for K-Means and Hierarchical.",
            )
            if n_clusters > 0:
                out["clustering_kwargs"]["n_clusters"] = int(n_clusters)
            else:
                out["clustering_kwargs"].pop("n_clusters", None)
        with col2:
            current_range = out["clustering_kwargs"].get("k_range") or (2, 10)
            k_min, k_max = st.slider(
                "k_range for auto-K (silhouette search)",
                min_value=2,
                max_value=15,
                value=(int(current_range[0]), int(current_range[1])),
            )
            out["clustering_kwargs"]["k_range"] = (k_min, k_max)

        st.caption("DBSCAN")
        col1, col2 = st.columns(2)
        with col1:
            eps_val = st.number_input(
                "eps (0 = automatic via k-NN knee)",
                min_value=0.0,
                max_value=10.0,
                value=float(out["clustering_kwargs"].get("eps") or 0.0),
                step=0.05,
                format="%.3f",
            )
            if eps_val > 0:
                out["clustering_kwargs"]["eps"] = float(eps_val)
            else:
                out["clustering_kwargs"].pop("eps", None)
        with col2:
            out["clustering_kwargs"]["min_samples"] = int(
                st.number_input(
                    "min_samples",
                    min_value=2,
                    max_value=50,
                    value=int(out["clustering_kwargs"].get("min_samples") or 5),
                    step=1,
                )
            )

        st.caption("Hierarchical")
        out["clustering_kwargs"]["linkage"] = st.selectbox(
            "Linkage method",
            options=["ward", "complete", "average", "single"],
            index=_safe_index(
                ["ward", "complete", "average", "single"],
                out["clustering_kwargs"].get("linkage", "ward"),
            ),
        )

    # ---- Anomaly detection ----
    with st.expander("Anomaly detection parameters", expanded=False):
        n_neighbors = st.number_input(
            "n_neighbors (LOF, 0 = default)",
            min_value=0,
            max_value=100,
            value=int(out["anomaly_kwargs"].get("n_neighbors") or 0),
            step=1,
        )
        if n_neighbors > 0:
            out["anomaly_kwargs"]["n_neighbors"] = int(n_neighbors)
        else:
            out["anomaly_kwargs"].pop("n_neighbors", None)

        random_state = st.number_input(
            "random_state (Isolation Forest, -1 = none)",
            min_value=-1,
            max_value=10000,
            value=int(out["anomaly_kwargs"].get("random_state", 42)),
            step=1,
        )
        if random_state >= 0:
            out["anomaly_kwargs"]["random_state"] = int(random_state)
        else:
            out["anomaly_kwargs"].pop("random_state", None)

    return out


# ============================================================
# Helpers
# ============================================================
def _safe_index(options: list, value) -> int:
    """Return the index of `value` in `options`, falling back to 0."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return 0


def _settings_diff(old: dict, new: dict) -> dict:
    """Return only the keys whose value changed, as {key: (old, new)}."""
    diff = {}
    keys = set(old.keys()) | set(new.keys())
    for k in sorted(keys):
        if old.get(k) != new.get(k):
            diff[k] = (old.get(k), new.get(k))
    return diff


def _show_diff_table(diff: dict):
    """Display diff as a clean table with Parameter, Old Value, New Value columns."""
    import pandas as pd

    rows = []
    for key, (old, new) in diff.items():
        rows.append({
            "Parameter": key,
            "Old Value": str(old) if old is not None else "(none)",
            "New Value": str(new) if new is not None else "(none)",
        })
    
    diff_df = pd.DataFrame(rows)
    st.dataframe(diff_df, use_container_width=True, hide_index=True)


def _rerun_pipeline(ctx: dict, settings: dict):
    """Re-execute AEDAPipeline.run() with new settings on the existing data."""
    from aeda.pipeline.runner import AEDAPipeline

    progress = st.progress(0, text="Initializing pipeline...")

    try:
        pipeline = AEDAPipeline(
            scale_method=settings.get("scale_method", "auto"),
            impute_strategy=settings.get("impute_strategy", "auto"),
            dim_method=settings.get("dim_method", "auto"),
            clustering_method=settings.get("clustering_method", "auto"),
            anomaly_method=settings.get("anomaly_method", "auto"),
            correlation_method=settings.get("correlation_method", "compare"),
            apply_clr=settings.get("apply_clr", False),
            contamination=float(settings.get("contamination", 0.05)),
            run_interpretation=bool(settings.get("run_interpretation", True)),
            reference_element=settings.get("reference_element", "Al"),
            baseline_strategy=settings.get("baseline_strategy", "deepest"),
            custom_baseline=settings.get("custom_baseline"),
            dim_kwargs=settings.get("dim_kwargs") or None,
            clustering_kwargs=settings.get("clustering_kwargs") or None,
            anomaly_kwargs=settings.get("anomaly_kwargs") or None,
        )

        progress.progress(20, text="Running analysis with the new settings...")
        time.sleep(0.2)

        results = pipeline.run(
            ctx["tmp_path"],
            exclude_cols=ctx.get("exclude_cols"),
            sheet_name=ctx.get("sheet_name"),
        )

        progress.progress(90, text="Storing new results...")

        # Update session state with new results and the settings used for this run.
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.run_context = {
            "tmp_path": ctx["tmp_path"],
            "sheet_name": ctx.get("sheet_name"),
            "exclude_cols": ctx.get("exclude_cols"),
            # Persist the effective settings actually resolved by the pipeline,
            # not what the user typed — that way the next page load reflects
            # what really ran (e.g. apply_clr=True when the plan recommended it).
            "settings": results.effective_settings or settings,
        }

        progress.progress(100, text="Done!")
        time.sleep(0.4)
        progress.empty()
        st.success(
            "Pipeline re-executed successfully. "
            "New results are now visible in Results, Audit and the other pages."
        )

        # Quick summary
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Variables analyzed",
            results.processed_data.shape[1] if results.processed_data is not None else "—",
        )
        if results.clustering:
            col2.metric("Clusters", results.clustering.n_clusters)
        if results.anomalies:
            col3.metric("Anomalies", results.anomalies.n_anomalies)

    except Exception as e:
        progress.empty()
        from app.components.errors import show_error
        show_error(
            "The pipeline could not complete with these settings. Please review the parameter configuration.",
            exc=e,
        )
