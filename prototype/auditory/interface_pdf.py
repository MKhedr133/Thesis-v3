"""Assemble questionnaire PDFs from one completed local interface export."""
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PDF_PAGES = {
    "shap": ("SHAP.pdf", (
        "01-overview.png",
        "02-comparison-all-difficulty-levels.png",
        "02-comparison-level-0.png",
        "02-comparison-level-1.png",
        "02-comparison-level-2.png",
        "02-comparison-level-3.png",
        "03-measurements-and-decision.png",
    )),
    "mamdani": ("Decision tree & Mamdani.pdf", (
        "01-overview-and-tree.png",
        "02-rule-activations-all-expanded.png",
        "03-fuzzy-output.png",
        "04-measurements-and-decision.png",
    )),
    "sugeno": ("Fuzzy C-means & Sugeno.pdf", (
        "01-overview-and-clusters.png",
        "01-clusters-all-expanded.png",
        "02-calculation.png",
        "03-measurements-and-decision.png",
    )),
}
PAGE_WIDTH = 842  # A4 landscape width in points; height follows the image ratio.


def write_interface_pdf(folder: Path, method: str) -> Path:
    """Write ordered, full-resolution images without cropping or resampling."""
    filename, names = PDF_PAGES[method]
    images = [folder / name for name in names]
    if method == "mamdani" and not images[1].is_file():
        # The exporter omits the expanded capture when no rules are activated.
        images[1] = folder / "02-rule-activations.png"
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Missing required PDF image: {image}")

    destination = folder.parent / filename
    temporary = destination.with_suffix(".pdf.tmp")
    try:
        pdf = canvas.Canvas(str(temporary), pageCompression=1)
        pdf.setTitle(destination.stem)
        for image in images:
            reader = ImageReader(str(image))
            width, height = reader.getSize()
            page_height = PAGE_WIDTH * height / width
            pdf.setPageSize((PAGE_WIDTH, page_height))
            pdf.drawImage(reader, 0, 0, width=PAGE_WIDTH, height=page_height, mask="auto")
            pdf.showPage()
        pdf.save()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
