"""
Gera o relatório PDF a partir do arquivo Markdown aprovado.
Usa markdown + weasyprint para gerar um PDF com estilo profissional.
"""
import markdown
from weasyprint import HTML
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MD_PATH  = os.path.join(BASE_DIR, "relatorio_titanic_iteracoes.md")
PDF_PATH = os.path.join(BASE_DIR, "Relatorio_Iteracoes_Titanic.pdf")

# ── 1. Ler o Markdown ────────────────────────────────────────────────────────
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_content = f.read()

# ── 2. Converter Markdown → HTML ─────────────────────────────────────────────
html_body = markdown.markdown(
    md_content,
    extensions=["tables", "fenced_code", "codehilite", "toc"],
    extension_configs={"codehilite": {"linenums": False, "css_class": "code"}}
)

# ── 3. Template HTML com CSS profissional ─────────────────────────────────────
html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<style>
  @page {{
    size: A4;
    margin: 2.2cm 2cm 2.5cm 2cm;
    @bottom-center {{
      content: "Página " counter(page) " de " counter(pages);
      font-size: 9px;
      color: #888;
      font-family: 'Segoe UI', Arial, sans-serif;
    }}
  }}

  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
  }}

  h1 {{
    font-size: 22pt;
    color: #1a3a5c;
    border-bottom: 3px solid #1a3a5c;
    padding-bottom: 10px;
    margin-top: 0;
    margin-bottom: 5px;
  }}

  h1 + h2:first-of-type {{
    margin-top: 5px;
    font-size: 13pt;
    color: #4a6a8c;
    font-weight: 400;
    border-bottom: none;
    padding-bottom: 0;
  }}

  h2 {{
    font-size: 15pt;
    color: #1a3a5c;
    border-bottom: 1.5px solid #ccd6e0;
    padding-bottom: 5px;
    margin-top: 30px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 12pt;
    color: #2c5282;
    margin-top: 20px;
    page-break-after: avoid;
  }}

  p {{
    text-align: justify;
    margin-bottom: 8px;
  }}

  strong {{
    color: #1a3a5c;
  }}

  em {{
    color: #555;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 10pt;
    page-break-inside: avoid;
  }}

  thead {{
    background-color: #1a3a5c;
    color: white;
  }}

  th {{
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    border: 1px solid #1a3a5c;
  }}

  td {{
    padding: 7px 10px;
    border: 1px solid #ddd;
  }}

  tbody tr:nth-child(even) {{
    background-color: #f0f4f8;
  }}

  tbody tr:hover {{
    background-color: #e2e8f0;
  }}

  code {{
    background-color: #eef2f7;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9.5pt;
    color: #c7254e;
  }}

  pre {{
    background-color: #1e293b;
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 9pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 12px 0;
  }}

  pre code {{
    background: none;
    color: #e2e8f0;
    padding: 0;
  }}

  hr {{
    border: none;
    border-top: 1.5px solid #ccd6e0;
    margin: 25px 0;
  }}

  ul, ol {{
    margin-top: 6px;
    margin-bottom: 12px;
    padding-left: 22px;
  }}

  li {{
    margin-bottom: 6px;
    line-height: 1.6;
  }}

  /* Emoji styling - mais sutil */
  .ranking-gold {{ color: #d4a017; }}

  blockquote {{
    border-left: 4px solid #3182ce;
    background-color: #ebf8ff;
    padding: 10px 15px;
    margin: 15px 0;
    border-radius: 0 6px 6px 0;
    font-style: normal;
  }}

  blockquote p {{
    margin: 5px 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# ── 4. Gerar PDF ──────────────────────────────────────────────────────────────
print(f"Gerando PDF...")
HTML(string=html_template).write_pdf(PDF_PATH)
print(f"✅ PDF gerado com sucesso: {PDF_PATH}")
print(f"   Tamanho: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")
