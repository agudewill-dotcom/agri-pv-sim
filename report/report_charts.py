import io
from reportlab.platypus import Image

def export_plotly_to_image(fig, width=500, height=300, pdf_width=None, pdf_height=None):
    """
    Exports a Plotly figure to a static PNG Image object for ReportLab.
    Uses kaleido engine internally via plotly.write_image.
    pdf_width/pdf_height override the ReportLab Image dimensions if provided.
    """
    if fig is None:
        return None
    
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2, engine="kaleido")
    img_buffer = io.BytesIO(img_bytes)
    
    if pdf_width is not None and pdf_height is not None:
        return Image(img_buffer, width=pdf_width, height=pdf_height)
    
    # Calculate aspect ratio
    aspect = height / width
    # Target width in reportlab points (A4 width is 595.27, margins are usually 50 on each side)
    target_width = 450
    target_height = target_width * aspect
    
    return Image(img_buffer, width=target_width, height=target_height)

def create_placeholder_image(width=450, height=200, text="Chart Placeholder"):
    """
    Creates a simple placeholder image if chart generation fails.
    """
    # For now, just return a small spacer, or we could generate a PIL image.
    from reportlab.platypus import Spacer
    return Spacer(width, height)

import matplotlib.pyplot as plt

def render_latex_to_image(latex_str, fontsize=12):
    """Renders a LaTeX string to a ReportLab Image via Matplotlib."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f'${latex_str}$', fontsize=fontsize)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=300)
    plt.close(fig)
    
    from PIL import Image as PILImage
    img = PILImage.open(buf)
    w, h = img.size
    w_pt = w * 72 / 300
    h_pt = h * 72 / 300
    
    buf.seek(0)
    return Image(buf, width=w_pt, height=h_pt)

