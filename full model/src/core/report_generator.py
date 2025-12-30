from fpdf import FPDF
import datetime
import json

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Inspection Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def generate_report(inspection, wagons):
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # Inspection Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Inspection ID: #{inspection['id']}", 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Video: {inspection['video_name']}", 0, 1)
    pdf.cell(0, 8, f"Date: {inspection['timestamp']}", 0, 1)
    pdf.cell(0, 8, f"Total Wagons: {inspection['total_wagons']}", 0, 1)
    pdf.ln(10)

    # Summary Stats & System Health (Side-by-Side)
    ocr_success = len([w for w in wagons if w['ocr_text'] and w['ocr_text'] != "OCR Failed"])
    defects = len([w for w in wagons if w['defects'] != "None"])
    night = len([w for w in wagons if w['is_night']])
    
    # Save current Y
    y_start = pdf.get_y()
    
    # LEFT COLUMN: Business Stats
    pdf.set_xy(10, y_start)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(90, 10, "Summary Statistics", 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(90, 8, f"Successful OCR: {ocr_success}", 0, 1)
    pdf.cell(90, 8, f"Wagons with Defects: {defects}", 0, 1)
    pdf.cell(90, 8, f"Night Conditions: {night}", 0, 1)
    
    # RIGHT COLUMN: System Health (Hardware)
    # Parse metrics
    fps = inspection.get('fps', 0.0)
    res = inspection.get('resolution', 'N/A')
    bright = inspection.get('avg_brightness', 0.0)
    blur_stats = inspection.get('blur_stats', '{}')
    try:
        blur_hist = json.loads(blur_stats)
    except:
        blur_hist = {}

    pdf.set_xy(110, y_start)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(90, 10, "System Health (Hardware)", 0, 1)
    
    pdf.set_xy(110, pdf.get_y())
    pdf.set_font('Arial', '', 10)
    pdf.cell(90, 6, f"Camera FPS: {fps:.1f}", 0, 1)
    
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(90, 6, f"Resolution: {res}", 0, 1)
    
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(90, 6, f"Avg Brightness: {bright:.1f} (0-255)", 0, 1)
    
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(90, 6, "Blur Severity Histogram:", 0, 1)
    
    # Mini histogram text
    current_y = pdf.get_y()
    pdf.set_xy(115, current_y)
    pdf.set_font('Courier', '', 9)
    # Sort keys for consistent order if standard bins
    bins = ['<50', '50-100', '100-200', '>200']
    for b in bins:
        count = blur_hist.get(b, 0)
        if count > 0:
            pdf.cell(80, 4, f"{b:<8}: {count}", 0, 1)
            pdf.set_x(115)
            
    # Reset cursor below both columns
    pdf.set_xy(10, y_start + 50)
    pdf.ln(5)

    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(20, 10, 'Index', 1)
    pdf.cell(50, 10, 'OCR Result', 1)
    pdf.cell(30, 10, 'Confidence', 1)
    pdf.cell(30, 10, 'Defects', 1)
    pdf.cell(40, 10, 'Timestamp', 1)
    pdf.ln()

    # Table Rows
    pdf.set_font('Arial', '', 10)
    for wagon in wagons:
        ocr_text = wagon['ocr_text'] or "N/A"
        conf = f"{wagon['ocr_confidence']*100:.1f}%" if wagon['ocr_confidence'] else "0%"
        defects = wagon['defects']
        ts = wagon['timestamp'].split(' ')[1] if ' ' in wagon['timestamp'] else wagon['timestamp']
        
        pdf.cell(20, 10, str(wagon['wagon_index']), 1)
        pdf.cell(50, 10, str(ocr_text), 1)
        pdf.cell(30, 10, conf, 1)
        pdf.cell(30, 10, str(defects), 1)
        pdf.cell(40, 10, ts, 1)
        pdf.ln()

    
    # -----------------------------
    # Visual Inspection Section
    # -----------------------------
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Detailed Visual Inspection", 0, 1, 'L')
    pdf.ln(5)
    
    import os
    
    for wagon in wagons:
        # Header for the Wagon
        pdf.set_font('Arial', 'B', 12)
        bg_color = (240, 240, 240)
        pdf.set_fill_color(*bg_color)
        title = f"Wagon #{wagon['wagon_index']} - OCR: {wagon['ocr_text'] or 'N/A'}"
        pdf.cell(0, 8, title, 0, 1, 'L', fill=True)
        pdf.ln(2)
        
        # Images Row
        # Calculate positions
        # Page width ~210mm. Margins ~10mm each. Usable ~190mm.
        # 3 Images -> ~60mm each with spacing.
        
        y_start = pdf.get_y()
        
        # Check if we have enough space for images (approx 50mm height needed), else new page
        if y_start > 230:
            pdf.add_page()
            y_start = pdf.get_y()
            pdf.cell(0, 8, title + " (Cont.)", 0, 1, 'L', fill=True)
            pdf.ln(2)
            y_start = pdf.get_y()
            
        img_width = 55
        img_height = 40 # Fixed height to keep alignment, or auto
        x_start = 10
        gap = 5
        
        # Define images to show
        images = [
            ("Original", wagon.get('original_image_path')),
            ("Deblurred", wagon.get('deblurred_image_path')),
            ("OCR Crop", wagon.get('cropped_number_path'))
        ]
        
        current_x = x_start
        max_h = 0
        
        for label, path in images:
            # Draw Label
            pdf.set_xy(current_x, y_start)
            pdf.set_font('Arial', 'I', 8)
            pdf.cell(img_width, 5, label, 0, 0, 'C')
            
            # Draw Image
            if path and os.path.exists(path):
                try:
                    pdf.image(path, x=current_x, y=y_start+6, w=img_width, h=img_height)
                except Exception as e:
                    pdf.set_xy(current_x, y_start + 20)
                    pdf.set_font('Arial', '', 8)
                    pdf.cell(img_width, 5, "[Image Error]", 0, 0, 'C')
            else:
                # Placeholder Box
                pdf.rect(current_x, y_start+6, img_width, img_height)
                pdf.set_xy(current_x, y_start+20)
                pdf.cell(img_width, 5, "No Image", 0, 0, 'C')
            
            current_x += (img_width + gap)
        
        pdf.set_y(y_start + img_height + 10)
        pdf.ln(2)

    return pdf
