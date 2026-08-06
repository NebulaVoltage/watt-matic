"""
SGCC EDA Executive PDF Report Generator (Part 7 & 8)
Generates `EDA_Report.pdf` using ReportLab with embedded high-DPI plots,
structured tables, key insights, ML recommendations, and presentation summaries.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)

class PDFReportGenerator:
    def __init__(self, loader, graphs_dir="graphs", output_pdf="EDA_Report.pdf"):
        self.loader = loader
        self.df = loader.df
        self.graphs_dir = graphs_dir
        self.output_pdf = output_pdf
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Sets up executive typography and colors."""
        self.title_style = ParagraphStyle(
            'DocTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0d3b66'),
            alignment=1, # Center
            spaceAfter=15
        )
        self.subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#555555'),
            alignment=1,
            spaceAfter=25
        )
        self.section_heading = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=15,
            leading=18,
            textColor=colors.HexColor('#0d3b66'),
            spaceBefore=15,
            spaceAfter=10,
            keepWithNext=True
        )
        self.body_style = ParagraphStyle(
            'BodyDark',
            parent=self.styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#222222'),
            spaceAfter=8
        )
        self.table_header_style = ParagraphStyle(
            'TableHeader',
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=colors.white,
            alignment=1
        )
        self.table_cell_style = ParagraphStyle(
            'TableCell',
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#222222')
        )

    def generate_pdf(self):
        """Builds the complete multi-page PDF document."""
        print(f"[INFO] Compiling presentation-ready PDF report '{self.output_pdf}'...")
        doc = SimpleDocTemplate(
            self.output_pdf,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        story = []

        # Document Header
        story.append(Paragraph("SGCC Electricity Theft Detection Dataset", self.title_style))
        story.append(Paragraph("Comprehensive Industry-Grade Exploratory Data Analysis & Analytics Report", self.subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0d3b66'), spaceAfter=15))

        # Executive Summary Section
        story.append(Paragraph("Executive Summary & Presentation Takeaways", self.section_heading))
        exec_text = (
            "This report delivers an end-to-end exploratory data analysis of the State Grid Corporation of China (SGCC) "
            "Electricity Theft Detection Dataset comprising <b>42,372 customer accounts</b> monitored over <b>1,034 days</b>. "
            "The study identifies key non-technical loss (NTL) patterns, class imbalance challenges (~9% theft incidence), "
            "data quality anomalies, and formulates an actionable roadmap for machine learning deployment."
        )
        story.append(Paragraph(exec_text, self.body_style))
        story.append(Spacer(1, 10))

        # Part 1: Dataset Overview Table
        story.append(Paragraph("Part 1 — Dataset Overview", self.section_heading))
        overview_data = [
            [Paragraph("<b>Metric</b>", self.table_header_style), Paragraph("<b>Value</b>", self.table_header_style)],
            [Paragraph("Total Customers (Rows)", self.table_cell_style), Paragraph(f"{len(self.df):,}", self.table_cell_style)],
            [Paragraph("Total Features (Columns)", self.table_cell_style), Paragraph(f"{self.df.shape[1]:,}", self.table_cell_style)],
            [Paragraph("Daily Consumption Features", self.table_cell_style), Paragraph(f"{len(self.loader.date_cols):,}", self.table_cell_style)],
            [Paragraph("Normal Customers (FLAG=0)", self.table_cell_style), Paragraph(f"{(self.df[self.loader.target_col]==0).sum():,} ({((self.df[self.loader.target_col]==0).sum()/len(self.df))*100:.1f}%)", self.table_cell_style)],
            [Paragraph("Theft Customers (FLAG=1)", self.table_cell_style), Paragraph(f"{(self.df[self.loader.target_col]==1).sum():,} ({((self.df[self.loader.target_col]==1).sum()/len(self.df))*100:.1f}%)", self.table_cell_style)],
            [Paragraph("Overall Missing Data Rate", self.table_cell_style), Paragraph(f"{(self.df[self.loader.date_cols].isna().sum().sum()/self.df[self.loader.date_cols].size)*100:.2f}%", self.table_cell_style)],
        ]
        t_overview = Table(overview_data, colWidths=[250, 250])
        t_overview.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d3b66')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_overview)
        story.append(Spacer(1, 15))

        # Part 3 Visualizations Showcase
        story.append(Paragraph("Part 3 — Selected Key Visualizations", self.section_heading))
        
        # Helper to embed images cleanly
        def embed_image(img_filename, caption_text):
            img_path = os.path.join(self.graphs_dir, img_filename)
            if os.path.exists(img_path):
                img = Image(img_path, width=500, height=260)
                cap = Paragraph(f"<b>Figure:</b> {caption_text}", self.body_style)
                return KeepTogether([img, cap, Spacer(1, 12)])
            return Paragraph(f"[Image {img_filename} not found]", self.body_style)

        story.append(embed_image("01_class_distribution.png", "Class Imbalance Distribution (Normal vs Theft)"))
        story.append(embed_image("03_average_daily_consumption.png", "Average Daily Electricity Consumption Profiles"))
        story.append(embed_image("12_compare_normal_vs_theft_customer.png", "Time Series Comparison: Normal Profile vs Meter Bypass Theft Pattern"))
        story.append(embed_image("18_zero_consumption_distribution.png", "Zero-Consumption Reading Distribution by Customer Class"))
        story.append(embed_image("20_monthly_consumption_trend.png", "Monthly Aggregated Consumption Trend (Seasonal Peak Shift)"))

        story.append(PageBreak())

        # Part 5 & 6: ML & Feature Engineering
        story.append(Paragraph("Part 5 & 6 — Machine Learning Strategy & Feature Engineering", self.section_heading))
        ml_text = (
            "<b>Machine Learning Insights:</b><br/>"
            "• <b>Severe Class Imbalance (~1:10):</b> Requires Focal Loss, SMOTE, or GBDT <code>scale_pos_weight</code> adjustment.<br/>"
            "• <b>Sparsity & Zero Readings:</b> High incidence of 0 kWh readings signals active physical bypass or meter tampering.<br/>"
            "• <b>Recommended Architecture:</b> Gradient Boosted Decision Trees (XGBoost/LightGBM) on engineered features combined with 1D-CNN + LSTM sequence models for raw temporal data."
        )
        story.append(Paragraph(ml_text, self.body_style))
        story.append(Spacer(1, 10))

        # Part 7 Presentation Summary Table
        story.append(Paragraph("Part 7 — Presentation Summary (PPT Ready)", self.section_heading))
        ppt_data = [
            [Paragraph("<b>Category</b>", self.table_header_style), Paragraph("<b>Key Takeaways</b>", self.table_header_style)],
            [Paragraph("<b>Key Insights</b>", self.table_cell_style), Paragraph("Theft customers display sudden 50-90% load drops, zero-consumption streaks, and absence of seasonal peak variations.", self.table_cell_style)],
            [Paragraph("<b>Key Challenges</b>", self.table_cell_style), Paragraph("High noise, missing readings (~2%), severe class imbalance (9% positive cases), and feature autocorrelation across 1,034 days.", self.table_cell_style)],
            [Paragraph("<b>Industrial Advantages</b>", self.table_cell_style), Paragraph("Automated theft detection reduces non-technical utility grid losses, improves revenue collection, and enhances grid safety.", self.table_cell_style)],
            [Paragraph("<b>ML Opportunities</b>", self.table_cell_style), Paragraph("Unsupervised anomaly detection (Autoencoders, Isolation Forest) combined with supervised LightGBM classifiers.", self.table_cell_style)],
            [Paragraph("<b>Limitations</b>", self.table_cell_style), Paragraph("Lack of geographic, weather, and tariff metadata; seasonal anomalies may mimic legitimate industrial shutdowns.", self.table_cell_style)],
        ]
        t_ppt = Table(ppt_data, colWidths=[130, 370])
        t_ppt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d3b66')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_ppt)
        story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)
        print(f"[SUCCESS] PDF report compiled and saved as '{self.output_pdf}'.")
        return self.output_pdf

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    pdf_gen = PDFReportGenerator(loader)
    pdf_gen.generate_pdf()
