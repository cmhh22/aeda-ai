"""
Page 3: Results

The main results dashboard. Shows the 4 core plots:
1. PCA biplot (the headline figure)
2. Correlation heatmap with hierarchical clustering
3. Cluster scatter with optional ground-truth comparison
4. Feature importance (if available)

Each plot is interactive (Plotly) and can be exported.
"""

import streamlit as st


def render():
    st.header("Results")

    results = st.session_state.get("results")
    if results is None:
        st.info("Run an analysis first from the Upload page.")
        return

    raw_df = results.raw_data

    # Detect available categorical columns for coloring
    categorical_cols = raw_df.select_dtypes(exclude="number").columns.tolist()

    # ---- TAB LAYOUT ----
    tab_pca, tab_corr, tab_cluster, tab_anomaly = st.tabs([
        "PCA", "Correlations", "Clustering", "Anomalies"
    ])

    # ============================================================
    # TAB 1: PCA
    # ============================================================
    with tab_pca:
        if results.dim_reduction is None or results.dim_reduction.method != "PCA":
            st.warning("PCA was not executed or failed. Check the Analysis Plan for details.")
            return

        from aeda.viz.dimensionality import pca_biplot, pca_scree_plot

        st.subheader("PCA biplot")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            color_by = st.selectbox(
                "Color by", options=["None"] + categorical_cols,
                index=1 if categorical_cols else 0,
                key="pca_color",
            )
        with col2:
            n_loadings = st.slider("Loading arrows", min_value=5, max_value=25, value=12, key="pca_loadings")
        with col3:
            n_comp = results.dim_reduction.n_components_selected
            pc_x = st.selectbox("X axis", options=list(range(1, n_comp + 1)), index=0, key="pca_x")
            pc_y = st.selectbox("Y axis", options=list(range(1, n_comp + 1)), index=1, key="pca_y")

        fig = pca_biplot(
            results.dim_reduction,
            df=raw_df,
            color_by=color_by if color_by != "None" else None,
            top_n_loadings=n_loadings,
            pc_x=pc_x,
            pc_y=pc_y,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scree plot
        with st.expander("Scree plot (variance explained)"):
            fig_scree = pca_scree_plot(results.dim_reduction)
            st.plotly_chart(fig_scree, use_container_width=True)

        # Loadings table
        with st.expander("Loadings table"):
            loadings = results.dim_reduction.loadings
            if loadings is not None:
                st.dataframe(
                    loadings.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1),
                    use_container_width=True,
                    height=400,
                )

    # ============================================================
    # TAB 2: CORRELATIONS
    # ============================================================
    with tab_corr:
        if results.correlations is None:
            st.warning("Correlation analysis was not executed.")
            return

        from aeda.viz.correlations import correlation_heatmap, cross_correlation_heatmap

        st.subheader("Correlation matrix")

        col1, col2 = st.columns([1, 1])
        with col1:
            corr_method = st.selectbox(
                "Method",
                options=["pearson", "spearman"],
                index=0,
                key="corr_method",
            )
        with col2:
            reorder = st.checkbox("Cluster-reorder axes", value=True, key="corr_reorder")

        if isinstance(results.correlations, dict) and corr_method in results.correlations:
            corr_result = results.correlations[corr_method]
        elif isinstance(results.correlations, dict) and "pearson" in results.correlations:
            corr_result = results.correlations["pearson"]
        else:
            corr_result = results.correlations

        fig = correlation_heatmap(corr_result, reorder=reorder)
        st.plotly_chart(fig, use_container_width=True)

        # Significant pairs
        if hasattr(corr_result, "significant_pairs") and corr_result.significant_pairs:
            with st.expander(f"Significant correlations ({corr_result.n_strong} strong, {corr_result.n_moderate} moderate)"):
                import pandas as pd
                pairs_df = pd.DataFrame(corr_result.significant_pairs[:30])
                st.dataframe(pairs_df, use_container_width=True)

        # Nonlinear candidates
        if isinstance(results.correlations, dict):
            nonlinear = results.correlations.get("nonlinear_candidates", [])
            if nonlinear:
                with st.expander(f"Nonlinear relationship candidates ({len(nonlinear)})"):
                    import pandas as pd
                    nl_df = pd.DataFrame(nonlinear[:20])
                    st.dataframe(nl_df, use_container_width=True)
                    st.caption("These variable pairs have much higher Spearman than Pearson correlation, suggesting a nonlinear relationship.")

        # Cross-correlation
        plan = results.plan
        if plan and plan.profile.has_heavy_metals and plan.profile.has_granulometry:
            with st.expander("Heavy metals vs. grain size"):
                fig_cross = cross_correlation_heatmap(
                    raw_df,
                    group_a=plan.profile.heavy_metal_cols,
                    group_b=plan.profile.granulometry_cols,
                    method="spearman",
                )
                st.plotly_chart(fig_cross, use_container_width=True)

    # ============================================================
    # TAB 3: CLUSTERING
    # ============================================================
    with tab_cluster:
        if results.clustering is None:
            st.warning("Clustering was not executed.")
            return

        if results.dim_reduction is None:
            st.warning("Dimensionality reduction needed for cluster visualization.")
            return

        from aeda.viz.clustering import cluster_scatter, cluster_composition

        st.subheader("Cluster analysis")

        compare_col = st.selectbox(
            "Compare clusters with",
            options=["None"] + categorical_cols,
            index=1 if categorical_cols else 0,
            key="cluster_compare",
        )

        fig = cluster_scatter(
            results.clustering,
            results.dim_reduction,
            df=raw_df,
            compare_with=compare_col if compare_col != "None" else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster metrics
        metrics = results.clustering.metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", results.clustering.n_clusters)
        sil = metrics.get("silhouette")
        col2.metric("Silhouette score", f"{sil:.3f}" if sil else "—")
        db = metrics.get("davies_bouldin")
        col3.metric("Davies-Bouldin", f"{db:.3f}" if db else "—")

        # Composition chart
        if compare_col != "None":
            with st.expander("Cluster composition"):
                fig_comp = cluster_composition(results.clustering, raw_df, compare_col)
                st.plotly_chart(fig_comp, use_container_width=True)

        # Feature importance
        if results.feature_importance is not None:
            with st.expander("Variables that most discriminate between clusters"):
                import pandas as pd
                imp = results.feature_importance.importances
                imp_df = pd.DataFrame({"Variable": imp.index, "Importance": imp.values})
                st.bar_chart(imp_df.set_index("Variable").head(15))

    # ============================================================
    # TAB 4: ANOMALIES
    # ============================================================
    with tab_anomaly:
        if results.anomalies is None:
            st.warning("Anomaly detection was not executed.")
            return

        st.subheader("Anomaly detection")

        col1, col2 = st.columns(2)
        col1.metric("Method", results.anomalies.method)
        col2.metric("Anomalies detected", results.anomalies.n_anomalies)

        if results.anomalies.n_anomalies > 0 and results.dim_reduction is not None:
            import plotly.graph_objects as go
            from aeda.viz.base import apply_default_layout

            scores = results.dim_reduction.components
            x_col, y_col = scores.columns[0], scores.columns[1]
            is_anomaly = results.anomalies.is_anomaly

            fig = go.Figure()

            # Normal points
            normal_mask = ~is_anomaly
            fig.add_trace(go.Scatter(
                x=scores.loc[normal_mask, x_col],
                y=scores.loc[normal_mask, y_col],
                mode="markers", name="Normal",
                marker=dict(size=7, color="#2E4057", opacity=0.5),
            ))

            # Anomalous points
            anomaly_mask = is_anomaly
            fig.add_trace(go.Scatter(
                x=scores.loc[anomaly_mask, x_col],
                y=scores.loc[anomaly_mask, y_col],
                mode="markers", name="Anomaly",
                marker=dict(size=10, color="#A32D2D", symbol="diamond",
                            line=dict(width=1, color="white")),
            ))

            apply_default_layout(fig, title="Anomalies in PCA space")
            fig.update_xaxes(title=x_col)
            fig.update_yaxes(title=y_col)
            st.plotly_chart(fig, use_container_width=True)

        # Anomaly details
        if results.anomalies.n_anomalies > 0:
            with st.expander("Anomalous samples"):
                import pandas as pd
                anomaly_idx = results.anomalies.anomaly_indices
                if raw_df is not None and len(anomaly_idx) > 0:
                    anomaly_rows = raw_df.loc[anomaly_idx]
                    st.dataframe(anomaly_rows, use_container_width=True)
