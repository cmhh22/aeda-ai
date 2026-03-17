# ============================================================
# AEDA Framework - Configuración del Paquete
# ============================================================
from setuptools import setup, find_packages

setup(
    name="aeda-framework",
    version="1.0.0",
    description="Análisis Exploratorio de Datos Ambientales con IA - LEA-CEAC",
    author="LEA-CEAC",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyarrow>=16.0.0",
        "scipy>=1.13.0",
        "scikit-learn>=1.5.0",
        "pyod>=2.0.0",
        "missingno>=0.5.2",
        "miceforest>=5.0.2",
        "umap-learn>=0.5.6",
        "shap>=0.45.0",
        "lime>=0.2.0.1",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "plotly>=5.22.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
)
