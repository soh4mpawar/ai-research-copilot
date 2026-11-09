import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

docx_path = os.path.abspath("C:/Users/soham/OneDrive/Desktop/rag/AI Research Copilot - Teacher Submission.docx")
pdf_path = os.path.abspath("C:/Users/soham/OneDrive/Desktop/rag/AI Research Copilot - Teacher Submission.pdf")

print(f"Converting DOCX to PDF via Word COM automation...")
print(f"Source: {docx_path}")
print(f"Target: {pdf_path}")

try:
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(docx_path)
    # wdFormatPDF = 17
    doc.SaveAs(pdf_path, FileFormat=17)
    page_count = doc.ComputeStatistics(2) # 2 = wdStatisticPages
    doc.Close()
    word.Quit()
    print(f"SUCCESS: PDF exported successfully!")
    print(f"Total Page Count: {page_count} pages.")
except Exception as e:
    print(f"win32com failed: {e}. Trying PowerShell COM approach...")
    import subprocess
    ps_cmd = f"""
    $w = New-Object -ComObject Word.Application
    $w.Visible = $false
    $d = $w.Documents.Open('{docx_path}')
    $d.SaveAs([ref]'{pdf_path}', [ref]17)
    $pages = $d.ComputeStatistics(2)
    Write-Host "PAGES:$pages"
    $d.Close()
    $w.Quit()
    """
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    print("PowerShell Output:", res.stdout)
    print("PowerShell Error:", res.stderr)
    if os.path.exists(pdf_path):
        print(f"SUCCESS: PDF exported via PowerShell! Size: {os.path.getsize(pdf_path)} bytes.")
    else:
        print("PDF export failed.")
