"""Build an academic-style PDF of the SWOT proposal from the doc's Markdown export.

Title page, contents with page numbers (two passes), numbered sections, captioned tables
and figures, the research notes as Appendix A. Printed with Microsoft Edge (headless).
"""
import re
import subprocess
import unicodedata
from pathlib import Path

import pymupdf as fitz
import markdown

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
EXPORT = ROOT / "export"
MAIN_MD = EXPORT / "4f3dab81905bb01d4b1d671b9e3b5984.md"
NOTES_MD = EXPORT / "f55fe1ae76eb9c1de6909f7706e3846b.md"
MAP_PNG = EXPORT / "0fd0cc02002c836fe5a6a5955166786c.png"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_PDF = ROOT / "SWOT Tank Water Tracker for Sri Lanka.pdf"
DOC_TITLE = "SWOT Tank Water Tracker for Sri Lanka"

TABLE_CAPTIONS = [
    "Related studies and the gap each leaves for this project",
    "The 13 study tanks",
    "Data sources",
    "Results of the October 2026 feasibility check",
    "Evaluation plan",
    "Project scope",
    "Tools, equipment and data access",
    "Team roles",
    "Risks and mitigations",
]
NOTE_TABLE_CAPTIONS = [
    "Key facts to quote",
    "SWOT limits and how the project handles them",
    "Ready-made starting points",
    "Partners and data to request",
]
FIGURES = {
    "map": ("../export/" + MAP_PNG.name,
            "Study area east of Anuradhapura (80.40–80.80°E, 8.10–8.60°N): the five major reservoirs "
            "and the eight village tanks."),
    "pipeline": ("fig2_pipeline.svg",
                 "Method pipeline: SWOT builds each tank's height–area–volume curve; Sentinel-1 and "
                 "Sentinel-2 water maps then update storage every 5–12 days."),
    "timeline": ("fig3_timeline.svg",
                 "Project timeline, October 2026 to June 2027. Dates are assumed and will be adjusted to "
                 "faculty deadlines; the study area is confirmed by 15 December, after the tank name check."),
}

CSS = """
@page { size: A4; margin: 24mm 22mm 22mm 24mm;
  @bottom-center { content: counter(page); font-family: "Times New Roman", serif; font-size: 10pt; } }
@page :first { @bottom-center { content: ""; } }
html { font-family: "Times New Roman", Times, serif; font-size: 11.5pt; color: #000; background: #fff; }
body { margin: 0; line-height: 1.42; }
p { margin: 0 0 7pt; text-align: justify; }
a { color: #1d3a6e; text-decoration: none; }
h1, h2, h3 { font-weight: bold; break-after: avoid; page-break-after: avoid; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt; }
h2 .num, h1 .num { display: inline-block; min-width: 26pt; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 2.5pt; text-align: left; }
li p { margin: 0; text-align: left; }
p.sub { break-after: avoid; page-break-after: avoid; }
table { width: 100%; border-collapse: collapse; margin: 2pt 0 11pt; font-size: 9.8pt; line-height: 1.28; }
th, td { border: 0.6pt solid #000; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #e9e9e9; font-weight: bold; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
.tcap { font-size: 10pt; margin: 9pt 0 3pt; text-align: left; break-after: avoid; page-break-after: avoid; }
figure { margin: 8pt 0 12pt; text-align: center; break-inside: avoid; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 175mm; }
figure.map img { max-height: 168mm; }
figcaption { font-size: 10pt; margin-top: 5pt; text-align: center; padding: 0 6mm; }
.title-page { height: 245mm; display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center; page-break-after: always; break-after: page; }
.title-page .uni { font-size: 14pt; letter-spacing: 1.5pt; text-transform: uppercase; }
.title-page .course { font-size: 12.5pt; margin-top: 6pt; }
.title-page .kind { font-size: 12pt; margin-top: 40pt; letter-spacing: 1pt; text-transform: uppercase; }
.title-page h1 { font-size: 24pt; margin: 12pt 0 10pt; }
.title-page .subtitle { font-size: 12.5pt; font-style: italic; max-width: 130mm; }
.title-page .group { font-size: 13pt; margin-top: 46pt; font-weight: bold; }
.title-page .sup { font-size: 11.5pt; margin-top: 8pt; }
.title-page .date { font-size: 11.5pt; margin-top: 30pt; }
.toc { page-break-after: always; break-after: page; }
.toc h1 { font-size: 16pt; margin: 0 0 14pt; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 0 0 6pt; }
.toc a { color: #000; display: flex; align-items: baseline; }
.toc .tn { min-width: 26pt; }
.toc .dots { flex: 1; border-bottom: 1pt dotted #000; margin: 0 5pt 3pt; }
.appendix { page-break-before: always; break-before: page; }
.appendix h1 { font-size: 16pt; margin: 0 0 6pt; }
ul.refs { list-style: none; padding-left: 0; }
ul.refs li { padding-left: 18pt; text-indent: -18pt; margin-bottom: 4pt; }
"""


def clean_main(md):
    md = md.replace("\u200b", "")
    md = md.split("## Introduction", 1)[1]
    md = "## Introduction" + md
    md = re.sub(r"\n\s*Research notes and extra sources:.*$", "\n", md.strip())
    md = re.sub(r"^&#91;image:.*$", "@@FIG:map@@", md, flags=re.M)
    md = re.sub(r"^&#91;embedded content: method pipeline.*$", "@@FIG:pipeline@@", md, flags=re.M)
    md = re.sub(r"^&#91;embedded content: project timeline.*$", "@@FIG:timeline@@", md, flags=re.M)
    md = re.sub(r"^\* ", "- ", md, flags=re.M)
    assert "&#91;" not in md, "unhandled placeholder"
    return md


def clean_notes(md):
    md = md.replace("\u200b", "")
    md = re.sub(r"^# Research notes\s*\n", "", md)
    md = re.sub(r"^- \[ \] ", "- ☐ ", md, flags=re.M)
    return md


def to_html(md):
    return markdown.markdown(md, extensions=["tables", "sane_lists"], output_format="html")


def caption_tables(body, captions, prefix):
    parts = body.split("<table>")
    assert len(parts) - 1 == len(captions), f"{len(parts) - 1} tables, {len(captions)} captions"
    out = parts[0]
    for i, rest in enumerate(parts[1:]):
        out += f'<p class="tcap"><b>Table {prefix}{i + 1}.</b> {captions[i]}</p><table>{rest}'
    return out


def sort_refs(html):
    # Each reference list A–Z by first author, whatever order the doc has.
    def sort_list(m):
        items = [re.sub(r"^\s*<p>(.*)</p>\s*$", r"\1", it, flags=re.S)
                 for it in re.findall(r"<li>(.*?)</li>", m.group(1), flags=re.S)]
        items.sort(key=lambda it: unicodedata.normalize("NFKD", re.sub(r"<[^>]+>", "", it)).casefold())
        return '<ul class="refs">\n' + "\n".join(f"<li>{it}</li>" for it in items) + "\n</ul>"
    return re.sub(r'<ul class="refs">(.*?)</ul>', sort_list, html, flags=re.S)


def build_html(pages=None):
    main = to_html(clean_main(MAIN_MD.read_text(encoding="utf-8")))
    for dup in ("<p><strong>The 13 study tanks</strong></p>", "<p><strong>Data</strong></p>"):
        assert dup in main, dup
        main = main.replace(dup, "")
    main = caption_tables(main, TABLE_CAPTIONS, "")
    main = re.sub(r"<p><strong>([^<]*)</strong></p>", r'<p class="sub"><strong>\1</strong></p>', main)
    # The map is too tall for the space left under Table 2, so it goes just above that table.
    map_slot, table2 = "<p>@@FIG:map@@</p>", '<p class="tcap"><b>Table 2.</b>'
    assert main.count(map_slot) == 1 and main.count(table2) == 1
    main = main.replace(map_slot, "").replace(table2, map_slot + table2)
    for n, key in enumerate(("map", "pipeline", "timeline"), start=1):
        src, cap = FIGURES[key]
        fig = f'<figure class="{key}"><img src="{src}" alt=""><figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>'
        main = main.replace(f"<p>@@FIG:{key}@@</p>", fig)
    assert "@@FIG" not in main

    toc, n = [], 0

    def number_h2(m):
        nonlocal n
        title = m.group(1)
        if title == "References":
            toc.append(("references", "", title))
            return '<h2 id="references">References</h2>'
        n += 1
        toc.append((f"s{n}", str(n), title))
        return f'<h2 id="s{n}"><span class="num">{n}</span>{title}</h2>'

    main = re.sub(r"<h2>(.*?)</h2>", number_h2, main)
    head, refs = main.split('<h2 id="references">', 1)
    main = head + '<h2 id="references">' + sort_refs(refs.replace("<ul>", '<ul class="refs">'))

    notes = to_html(clean_notes(NOTES_MD.read_text(encoding="utf-8")))
    notes = caption_tables(notes, NOTE_TABLE_CAPTIONS, "A")
    k = 0

    def number_note(m):
        nonlocal k
        if m.group(1) == "Sources":
            return '<h2 id="a-sources">Sources</h2>'
        k += 1
        return f'<h2 id="a{k}"><span class="num">A.{k}</span>{m.group(1)}</h2>'

    notes = re.sub(r"<h2>(.*?)</h2>", number_note, notes)
    notes = notes.replace("<h2 id=\"a-sources\">Sources</h2>\n<ul>", "<h2 id=\"a-sources\">Sources</h2>\n<ul class=\"refs\">")
    toc.append(("appendix-a", "", "Appendix A. Research notes"))

    pages = pages or {}
    toc_items = "".join(
        f'<li><a href="#{i}"><span class="tn">{num}</span><span class="tt">{t}</span>'
        f'<span class="dots"></span><span class="pg">{pages.get(i, "")}</span></a></li>'
        for i, num, t in toc)
    title_page = f"""
<section class="title-page">
  <div class="uni">University of Moratuwa</div>
  <div class="course">IS4990 Comprehensive Group Project</div>
  <div class="kind">Project proposal</div>
  <h1>{DOC_TITLE}</h1>
  <div class="subtitle">An AI system that estimates how much water each tank holds, using free satellite data</div>
  <div class="group">Group 6 · JARVIS</div>
  <div class="sup">Supervisors: Dr. Upeksha Ganegoda (IT) · Prof. (Mrs) Thanuja Sandanayake (IDS)</div>
  <div class="date">29 October 2026</div>
</section>"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{DOC_TITLE}: project proposal (draft)</title><style>{CSS}</style></head><body>
{title_page}
<section class="toc"><h1>Contents</h1><ol>{toc_items}</ol></section>
<main>{main}</main>
<section class="appendix" id="appendix-a"><h1>Appendix A. Research notes</h1>{notes}</section>
</body></html>"""
    path = BUILD / "proposal.html"
    path.write_text(doc, encoding="utf-8")
    return path, toc


def render(html_path, pdf_path):
    profile = BUILD / "edge-profile"
    cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
           f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", "--virtual-time-budget=10000",
           html_path.as_uri()]
    if pdf_path.exists():
        pdf_path.unlink()
    subprocess.run(cmd, check=True, timeout=240, capture_output=True)
    assert pdf_path.exists(), "Edge did not write the PDF"


def find_pages(pdf_path, toc):
    doc = fitz.open(pdf_path)
    found = {}
    for idx, num, title in toc:
        target = title.split(". ", 1)[-1] if idx == "appendix-a" else title
        for pno in range(2, len(doc)):
            spans = [s for b in doc[pno].get_text("dict")["blocks"] for l in b.get("lines", [])
                     for s in l["spans"] if s["size"] >= 13]
            if any(target in s["text"] for s in spans):
                found[idx] = pno + 1
                break
    return found, len(doc)


def main():
    html_path, toc = build_html()
    tmp_pdf = BUILD / "pass1.pdf"
    render(html_path, tmp_pdf)
    pages, n1 = find_pages(tmp_pdf, toc)
    missing = [t for i, _, t in toc if i not in pages]
    print("pass 1:", n1, "pages; missing in contents:", missing)
    html_path, toc = build_html(pages)
    render(html_path, OUT_PDF)
    pages2, n2 = find_pages(OUT_PDF, toc)
    print("final:", n2, "pages; contents numbers stable:", pages2 == pages)
    print({t: pages.get(i) for i, _, t in toc})


if __name__ == "__main__":
    main()
