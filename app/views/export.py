"""
Page: Export

A simple download center. Offers a single Excel workbook with every result
table (one sheet each) plus per-table CSV downloads. Requested by the
scientific tutor (LEA-CEAC) to take the tables into external tools.
"""

import streamlit as st

from app.i18n import t
from app.exports import collect_tables, to_excel_bytes


def render():
    from app.components.page_header import page_header

    page_header(
        title=t("Export"),
        description=t("Download the result tables as Excel or CSV."),
        icon=":material/download:",
    )

    results = st.session_state.get("results")
    if results is None:
        st.info(t("Run an analysis first from the Upload page."))
        return

    tables = collect_tables(results)
    if not tables:
        st.info(t("No tables available to export yet."))
        return

    # ---- PDF report (decisions + validation + results + interpretation) ----
    st.subheader(t("Analysis report (PDF)"))
    st.caption(t("A readable report: decisions, validation, key results, interpretation and parameters."))
    try:
        from app.report import build_report_pdf
        pdf = build_report_pdf(results, filename=st.session_state.get("filename"))
        st.download_button(
            t("Download report (.pdf)"),
            data=pdf,
            file_name="aeda_informe.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    except Exception as e:
        from app.components.errors import show_error
        show_error(t("Could not build the PDF report."), exc=e)

    st.divider()

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
            use_container_width=True,
        )
    except Exception as e:
        from app.components.errors import show_error
        show_error(t("Could not build the Excel workbook."), exc=e)

    st.caption(
        t(
            "Tip: to download a single table on its own, hover over it on the "
            "Results or Audit page and use the table's download button."
        )
    )

    st.divider()
    st.caption(t("Tables included: {names}").format(names=", ".join(tables.keys())))
