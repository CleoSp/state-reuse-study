"""Generate charts from verified stream-training measurements."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor

COLORS = [HexColor(c) for c in ("#1B365D","#C25431","#00806C","#8264A8")]


def base(title: str, subtitle: str) -> Drawing:
    drawing = Drawing(600,350)
    drawing.add(String(48,326,title,fontName="Helvetica-Bold",fontSize=15,fillColor=COLORS[0]))
    drawing.add(String(48,306,subtitle,fontName="Helvetica",fontSize=9))
    return drawing


def legend(drawing: Drawing, labels: list[str]) -> None:
    for i,label in enumerate(labels):
        x = 48+i*138
        drawing.add(Line(x,25,x+20,25,strokeColor=COLORS[i],strokeWidth=2))
        drawing.add(String(x+25,21,label,fontName="Helvetica",fontSize=9))


def main() -> None:
    source = Path("reports/stream_results.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["provenance"]["synthetic"] or not all(data[f]["diagnostics"]["verified"] for f in ("maze","circuit")):
        raise ValueError("verified empirical data required")
    out = Path("reports/figures/prompt05b")
    out.mkdir(parents=True,exist_ok=True)
    manifest = []
    for family in ("maze","circuit","circuit-depth"):
        frames = ([r for r in data["addenda"]["depth"]["per_frame"] if r["primary"]]
                  if family == "circuit-depth" else data[family]["per_frame"])
        arms = ["restart","carry","spatial_gate","answer_only"]
        for k in sorted({r["K"] for r in frames}):
            drawing = base(f"{family.capitalize()}: accuracy across edits, K={k}",
                "Three training seeds averaged; 256 paired validation roots. Frame zero is the initial solve.")
            plot = LinePlot()
            plot.x,plot.y,plot.width,plot.height = 55,65,505,215
            plot.data = [[(f,mean(r["exact_accuracy"] for r in frames if r["arm"] == a and r["K"] == k and r["frame"] == f)) for f in range(33)] for a in arms]
            plot.xValueAxis.valueMin,plot.xValueAxis.valueMax,plot.xValueAxis.valueStep = 0,32,8
            plot.yValueAxis.valueMin,plot.yValueAxis.valueMax,plot.yValueAxis.valueStep = 0,1,.2
            for i in range(len(arms)):
                plot.lines[i].strokeColor,plot.lines[i].strokeWidth = COLORS[i],1.8
            drawing.add(plot)
            drawing.add(String(268,43,"Edit frame",fontName="Helvetica",fontSize=9))
            legend(drawing,arms)
            name = f"{family}-decay-K{k}.svg"
            renderSVG.drawToFile(drawing,str(out/name))
            manifest.append(name)
        if family == "circuit-depth":
            continue
        histograms = data[family]["diagnostics"]["retention_histograms"]
        for group in ("all","edited","distant"):
            records = [next(r for r in histograms if r["stratum"] == s and r["group"] == group) for s in ("low","high")]
            missing = [r["stratum"] for r in records if not r["nodes"]]
            records = [r for r in records if r["nodes"]]
            drawing = base(f"{family.capitalize()}: spatial retention, {group} nodes",
                "Missing strata: " + ", ".join(missing) if missing else
                "Low versus high impact; descriptive pooled nodes across three seeds, not independent replicates.")
            name = f"{family}-retention-{group}.svg"
            if not records:
                drawing.add(String(55,180,"No observed nodes in this group; no distribution estimated.",fontSize=12))
                renderSVG.drawToFile(drawing,str(out/name))
                manifest.append(name)
                continue
            chart = VerticalBarChart()
            chart.x,chart.y,chart.width,chart.height = 55,75,505,205
            chart.data = [tuple(n/r["nodes"] for n in r["counts"]) for r in records]
            chart.valueAxis.valueMin = 0
            chart.categoryAxis.categoryNames = [f"{(i+.5)*.05:.2f}" if i in (0,4,8,12,16,19) else "" for i in range(20)]
            chart.categoryAxis.labels.fontSize = 8
            for i in range(len(records)):
                chart.bars[i].fillColor = COLORS[i]
                chart.bars[i].strokeColor = COLORS[i]
            drawing.add(chart)
            drawing.add(String(185,48,"Mean a/z retention (bin centers; bin width 0.05)",fontName="Helvetica",fontSize=9))
            legend(drawing,[r["stratum"]+" impact" for r in records])
            renderSVG.drawToFile(drawing,str(out/name))
            manifest.append(name)
    import hashlib
    import reportlab
    (out / "manifest.json").write_text(json.dumps({"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"reportlab_version":reportlab.Version,
        "charts":{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in manifest}},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
