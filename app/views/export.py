"""
Page: Export

A simple download center. Offers a single Excel workbook with every result
table (one sheet each) plus per-table CSV downloads. Requested by the
scientific tutor (LEA-CEAC) to take the tables into external tools.
"""

import streamlit as st

from app.i18n import t
from app.exports import collect_tables, to_excel_bytes, to_csv_bytes, safe_filename


def render():
    from app.components.page_header import page_header

    page_header(
        title=t("Export"),
        description=t("Download the result tables as Excel or CSV."),
        icon="⬇️",
    )

    results = st.session_state.get("results")
    if results is None:
        st.info(t("Run an analysis first from the Upload page."))
        return

    tables = collect_tables(results)
    if not tables:
        st.info(t("No tables available to export yet."))
        return

    # ---- One workbook with everything ----
    st.subheader(t("All tables (Excel)"))
    st.caption(t("One workbook with every result table as a separate sheet."))
    try:
        xlsx = to_excel_bytes(tables)
        st.download_button(
            t("Download Excel (.xlsx)"),
            data=xlsx,
            file_name="aeda_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    except Exception as e:
        from app.components.errors import show_error
        show_error(t("Could not build the Excel workbook."), exc=e)

    st.divider()

    # ---- Individual CSVs ----
    st.subheader(t("Individual tables (CSV)"))
    for name, df in tables.items():
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{name}** · {df.shape[0]}×{df.shape[1]}")
        c2.download_button(
            t("CSV"),
            data=to_csv_bytes(df),
            file_name=safe_filename(name, "csv"),
            mime="text/csv",
            key=f"csv_{name}",
            use_container_width=True,
        )
