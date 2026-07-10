"""PDF report generation for traffic analysis results."""

from typing import Dict, Optional
from datetime import datetime

from fpdf import FPDF


class PDFReportGenerator:
    """Generates PDF reports with traffic analysis results."""

    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def generate_report(
        self,
        counts: Dict[str, int],
        density: str,
        duration_seconds: float,
        avg_fps: float,
        model_name: str,
        output_path: str,
        title: str = "Vehicle Count Analysis Report",
    ):
        """Generate a complete PDF report.

        Args:
            counts: Vehicle counts per class.
            density: Traffic density level string.
            duration_seconds: Video duration processed.
            avg_fps: Average processing FPS.
            model_name: Name of detection model used.
            output_path: Output PDF file path.
            title: Report title.
        """
        self.pdf.add_page()

        # Title
        self.pdf.set_font("Arial", "B", 20)
        self.pdf.cell(0, 15, title, ln=True, align="C")
        self.pdf.ln(5)

        # Metadata
        self.pdf.set_font("Arial", "", 10)
        self.pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        self.pdf.cell(0, 8, f"Model: {model_name}", ln=True)
        self.pdf.cell(0, 8, f"Duration: {duration_seconds:.1f} seconds", ln=True)
        self.pdf.cell(0, 8, f"Avg FPS: {avg_fps:.1f}", ln=True)
        self.pdf.ln(10)

        # Summary section
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Traffic Summary", ln=True)
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.ln(5)

        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, f"Traffic Density: {density}", ln=True)
        self.pdf.cell(0, 10, f"Total Vehicles Counted: {counts.get('total', 0)}", ln=True)
        self.pdf.ln(5)

        # Per-class counts table
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Vehicle Counts by Class", ln=True)
        self.pdf.ln(3)

        # Table header
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.set_fill_color(200, 220, 255)
        self.pdf.cell(95, 8, "Vehicle Class", border=1, fill=True, align="C")
        self.pdf.cell(95, 8, "Count", border=1, fill=True, align="C")
        self.pdf.ln()

        # Table rows
        self.pdf.set_font("Arial", "", 10)
        for class_name, count in sorted(counts.items()):
            if class_name == "total":
                continue
            self.pdf.cell(95, 8, class_name.replace("_", " ").title(), border=1, align="C")
            self.pdf.cell(95, 8, str(count), border=1, align="C")
            self.pdf.ln()

        # Total row
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.cell(95, 8, "TOTAL", border=1, fill=True, align="C")
        self.pdf.cell(95, 8, str(counts.get("total", 0)), border=1, fill=True, align="C")
        self.pdf.ln(15)

        # Footer
        self.pdf.set_font("Arial", "I", 8)
        self.pdf.cell(
            0, 10,
            "AI-Powered Vehicle Count Prediction for Smart City Traffic Management",
            ln=True, align="C",
        )

        # Save
        self.pdf.output(output_path)
