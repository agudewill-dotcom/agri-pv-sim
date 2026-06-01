from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import TableStyle

# Brand Colors
BRAND_BLUE = HexColor("#1e3a8a")  # Deep blue from logo
BRAND_LIGHT_BLUE = HexColor("#eff6ff")
BRAND_GRAY = HexColor("#475569")
BRAND_DARK = HexColor("#0f172a")
BRAND_GREEN = HexColor("#16a34a")
BRAND_RED = HexColor("#dc2626")

def get_report_styles():
    """Returns a dictionary of customized ReportLab paragraph styles."""
    styles = getSampleStyleSheet()
    
    # Base paragraph style
    base_style = ParagraphStyle(
        'BaseStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=BRAND_DARK,
        spaceAfter=10
    )
    
    custom_styles = {
        'Normal': base_style,
        'NormalGray': ParagraphStyle(
            'NormalGray',
            parent=base_style,
            textColor=BRAND_GRAY
        ),
        'Disclaimer': ParagraphStyle(
            'Disclaimer',
            parent=base_style,
            fontSize=8,
            leading=10,
            textColor=BRAND_GRAY,
            fontName='Helvetica-Oblique'
        ),
        'Title': ParagraphStyle(
            'Title',
            parent=base_style,
            fontName='Helvetica-Bold',
            fontSize=28,
            leading=34,
            textColor=BRAND_BLUE,
            spaceAfter=20,
            alignment=TA_LEFT
        ),
        'Heading1': ParagraphStyle(
            'Heading1',
            parent=base_style,
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=BRAND_BLUE,
            spaceBefore=20,
            spaceAfter=12,
            borderPadding=0
        ),
        'Heading2': ParagraphStyle(
            'Heading2',
            parent=base_style,
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=BRAND_DARK,
            spaceBefore=15,
            spaceAfter=8
        ),
        'Heading3': ParagraphStyle(
            'Heading3',
            parent=base_style,
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=BRAND_DARK,
            spaceBefore=12,
            spaceAfter=6
        ),
        'Formula': ParagraphStyle(
            'Formula',
            parent=base_style,
            fontName='Courier',
            fontSize=10,
            leading=14,
            textColor=HexColor("#334155"),
            leftIndent=20,
            spaceBefore=8,
            spaceAfter=8,
            backColor=BRAND_LIGHT_BLUE,
            borderPadding=5
        ),
        'Footer': ParagraphStyle(
            'Footer',
            parent=base_style,
            fontSize=8,
            textColor=BRAND_GRAY,
            alignment=TA_LEFT
        ),
        'FooterRight': ParagraphStyle(
            'FooterRight',
            parent=base_style,
            fontSize=8,
            textColor=BRAND_GRAY,
            alignment=TA_RIGHT
        ),
        'Bullet': ParagraphStyle(
            'Bullet',
            parent=base_style,
            leftIndent=20,
            firstLineIndent=-10,
            spaceBefore=2,
            spaceAfter=2
        )
    }
    return custom_styles

def get_table_style_standard():
    """Returns a standard professional table style."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#ffffff")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#ffffff")),
        ('TEXTCOLOR', (0, 1), (-1, -1), BRAND_DARK),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")])
    ])

def get_table_style_kpi():
    """Returns a table style designed for KPI blocks without vertical grid lines."""
    return TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (-1, -1), BRAND_DARK),
        ('TEXTCOLOR', (1, 0), (1, -1), BRAND_BLUE),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0"))
    ])
