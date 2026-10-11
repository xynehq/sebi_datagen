"""Install the OCR dataset pipeline. `pip install -e .` does not download weights."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def _requirements() -> list[str]:
    lines = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            lines.append(text)
    return lines


setup(
    name="indic-ocr-pipeline",
    version="0.1.0",
    description="Multilingual OCR dataset, training, and evaluation pipeline.",
    python_requires=">=3.11",
    install_requires=_requirements(),
    extras_require={
        "chitrapathak": ["transformers>=4.46", "peft>=0.13", "accelerate>=1.0"],
    },
    packages=(
        find_packages(where="src")
        + find_packages(include=["pipeline*", "evaluation*", "training*", "models*"])
        + find_packages(where="models", include=["IndicPhotoOCR*"])
    ),
    package_dir={
        "indic_ocr_pipeline": "src/indic_ocr_pipeline",
        "pipeline": "pipeline",
        "evaluation": "evaluation",
        "training": "training",
        "models": "models",
        "IndicPhotoOCR": "models/IndicPhotoOCR",
    },
    entry_points={"console_scripts": ["indic-ocr=pipeline.cli:main"]},
)
