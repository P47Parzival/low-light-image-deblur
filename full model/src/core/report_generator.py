from fpdf import FPDF
import datetime

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

    # Summary Stats
    ocr_success = len([w for w in wagons if w['ocr_text'] and w['ocr_text'] != "OCR Failed"])
    defects = len([w for w in wagons if w['defects'] != "None"])
    night = len([w for w in wagons if w['is_night']])
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Summary Statistics", 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Successful OCR: {ocr_success}", 0, 1)
    pdf.cell(0, 8, f"Wagons with Defects: {defects}", 0, 1)
    pdf.cell(0, 8, f"Night Conditions: {night}", 0, 1)
    pdf.ln(10)

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

    return pdf
