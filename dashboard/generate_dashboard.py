"""
Generate a self-contained HTML dashboard for a single clip.

Usage:
    python dashboard/generate_dashboard.py \
        --sequence_name SNMOT-148 \
        --output_html outputs/SNMOT-148/dashboard/index.html
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import numpy as np


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_dashboard_data(seq_name, base_dir):
    data = {
        "sequence_name": seq_name,
        "metadata": {
            "frames": 750,
            "fps": 25.0,
            "duration_s": 30.0,
        }
    }

    pred_metrics_path = os.path.join(base_dir, f"{seq_name}_player_physical_metrics.csv")
    pred_metrics = load_csv(pred_metrics_path)
    data["player_metrics"] = []
    for r in pred_metrics:
        data["player_metrics"].append({
            "track_id": int(r["track_id"]),
            "team_id": int(r["team_id"]),
            "role": r["role"],
            "minutes_visible": float(r["minutes_visible"]),
            "distance_m": float(r["distance_m"]),
            "max_speed_kmh": float(r["max_speed_kmh"]),
            "mean_speed_kmh": float(r["mean_speed_kmh"]),
            "sprint_count": int(r["sprint_count"]),
            "sprint_distance_m": float(r["sprint_distance_m"]),
            "high_speed_distance_m": float(r["high_speed_distance_m"]),
        })

    shape_path = os.path.join(base_dir, f"{seq_name}_team_shape_metrics.csv")
    shape_rows = load_csv(shape_path)
    data["team_shape"] = []
    for r in shape_rows:
        fid = int(r["frame_id"])
        if fid % 5 == 0:
            data["team_shape"].append({
                "frame_id": fid,
                "time_s": float(r["time_s"]),
                "team_id": int(r["team_id"]),
                "n_players_visible": int(r["n_players_visible"]),
                "centroid_x_m": float(r["centroid_x_m"]),
                "centroid_y_m": float(r["centroid_y_m"]),
                "width_m": float(r["width_m"]),
                "compactness": float(r["compactness"]) if r["compactness"] != '' else None,
            })

    comp_dir = os.path.join(base_dir, "comparison")
    frame_err_path = os.path.join(comp_dir, "frame_matches.csv")
    data["frame_errors"] = []
    if os.path.exists(frame_err_path):
        for r in load_csv(frame_err_path):
            data["frame_errors"].append({
                "frame_id": int(r["frame_id"]),
                "n_pred": int(r["n_pred"]),
                "n_gt": int(r["n_gt"]),
                "n_matched": int(r["n_matched"]),
                "mean_error_m": float(r["mean_error_m"]) if r["mean_error_m"] != '' else None,
                "max_error_m": float(r["max_error_m"]) if r["max_error_m"] != '' else None,
            })

    summary_path = os.path.join(comp_dir, "comparison_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            cs = json.load(f)
        data["comparison_summary"] = cs
        data["team_mapping"] = cs.get("team_mapping_pred_to_gt", {})
    else:
        data["comparison_summary"] = {}
        data["team_mapping"] = {}

    timeseries_path = os.path.join(base_dir, f"{seq_name}_player_physical_timeseries.csv")
    ts_rows = load_csv(timeseries_path)
    top5_ids = [p["track_id"] for p in sorted(data["player_metrics"], key=lambda x: x["distance_m"], reverse=True)[:5]]
    speed_series = defaultdict(lambda: defaultdict(float))
    for r in ts_rows:
        tid = int(r["track_id"])
        if tid in top5_ids:
            fid = int(r["frame_id"])
            speed_series[tid][fid] = float(r["speed_kmh"])

    data["speed_profiles"] = {}
    for tid in top5_ids:
        frames = sorted(speed_series[tid].keys())
        data["speed_profiles"][str(tid)] = [
            {"frame_id": fid, "time_s": round((fid - 1) / 25.0, 2), "speed_kmh": speed_series[tid][fid]}
            for fid in frames
        ]

    pred_tracking_path = os.path.join(base_dir, f"{seq_name}_tracking_2d.csv")
    gt_tracking_path = os.path.join(base_dir, f"{seq_name}_gt_tracking_2d.csv")

    def load_positions(path):
        out = defaultdict(list)
        for r in load_csv(path):
            if r.get("entity_type") != "player":
                continue
            if int(r.get("visible", 0)) == 0:
                continue
            team_id = int(r.get("team_id", -1))
            if team_id < 0:
                continue
            fid = int(r["frame_id"])
            out[fid].append({
                "track_id": int(r["track_id"]),
                "team_id": team_id,
                "x": float(r["x_m"]),
                "y": float(r["y_m"]),
            })
        return out

    pred_pos = load_positions(pred_tracking_path)
    gt_pos = load_positions(gt_tracking_path)

    all_frames = sorted(set(pred_pos.keys()) | set(gt_pos.keys()))
    data["positions"] = {}
    for fid in all_frames:
        data["positions"][str(fid)] = {
            "pred": pred_pos.get(fid, []),
            "gt": gt_pos.get(fid, []),
        }

    return data


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{SEQ_NAME}} — Tactical Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b0f19;--card:#111827;--text:#e2e8f0;--muted:#94a3b8;--accent:#3b82f6;--accent2:#10b981;--accent3:#f59e0b;--border:#1e293b;}
*{margin:0;padding:0;box-sizing:border-box;font-family:Inter,system-ui,sans-serif}
body{background:var(--bg);color:var(--text);padding:24px;}
h1{font-size:22px;margin-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}
.card h3{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:0.5px;margin-bottom:8px}
.card .val{font-size:28px;font-weight:700}
.card .sub{font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
th{text-align:left;padding:10px 12px;color:var(--muted);border-bottom:1px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:0.5px}
td{padding:10px 12px;border-bottom:1px solid var(--border)}
tr:hover td{background:rgba(255,255,255,0.03)}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px}
.chart-title{font-size:14px;font-weight:600;margin-bottom:12px}
canvas{max-height:280px}
.pitch-wrap{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
#pitchCanvas{background:#1a3c1a;border:1px solid var(--border);border-radius:8px;cursor:crosshair}
.controls{display:flex;flex-direction:column;gap:10px;min-width:220px}
input[type=range]{width:100%}
.footer{margin-top:24px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<h1>&#9917; {{SEQ_NAME}} — Tactical Dashboard</h1>
<div class="sub">{{DURATION}} · {{FRAMES}} frames @ {{FPS}} FPS · Action: Goal</div>

<div class="grid">
  <div class="card"><h3>Total Distance (Pred)</h3><div class="val" id="kpi-dist">--</div><div class="sub">metres</div></div>
  <div class="card"><h3>Max Speed (Pred)</h3><div class="val" id="kpi-maxsp">--</div><div class="sub">km/h</div></div>
  <div class="card"><h3>Sprints (Pred)</h3><div class="val" id="kpi-sprints">--</div><div class="sub">count</div></div>
  <div class="card"><h3>RMSE vs GT</h3><div class="val" id="kpi-rmse">--</div><div class="sub">metres (matches &le; 3m)</div></div>
  <div class="card"><h3>Coverage</h3><div class="val" id="kpi-cov">--</div><div class="sub">% GT matched</div></div>
  <div class="card"><h3>Precision</h3><div class="val" id="kpi-prec">--</div><div class="sub">% pred correct</div></div>
</div>

<div class="chart-card">
  <div class="chart-title">Players — Physical Summary</div>
  <div style="overflow-x:auto">
    <table id="playerTable">
      <thead>
        <tr><th>Track</th><th>Team</th><th>Role</th><th>Min Visible</th><th>Distance (m)</th><th>Max Spd (km/h)</th><th>Mean Spd</th><th>Sprints</th><th>Sprint Dist</th><th>Hi-Spd Dist</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px">
  <div class="chart-card"><div class="chart-title">Team Width Over Time</div><canvas id="widthChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Compactness Over Time</div><canvas id="compactChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Frame Error (Pred vs GT)</div><canvas id="errorChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Top 5 Players — Speed Profile</div><canvas id="speedChart"></canvas></div>
</div>

<div class="chart-card">
  <div class="chart-title">2D Pitch Viewer <span style="color:var(--muted);font-weight:400">(frame slider)</span></div>
  <div class="pitch-wrap">
    <canvas id="pitchCanvas" width="840" height="544"></canvas>
    <div class="controls">
      <label>Frame: <span id="frameLabel">1</span> / <span id="totalFrames">750</span></label>
      <input type="range" id="frameSlider" min="1" max="750" value="1" step="1">
      <div style="font-size:12px;color:var(--muted)">Time: <span id="timeLabel">0.00</span> s</div>
      <div style="margin-top:8px;font-size:12px">
        <div><span style="display:inline-block;width:10px;height:10px;background:#3b82f6;border-radius:50%"></span> Pred Team 0 &rarr; GT Team <span id="map0">?</span></div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#10b981;border-radius:50%"></span> Pred Team 1 &rarr; GT Team <span id="map1">?</span></div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#3b82f6;border:2px solid #fff"></span> GT Team 0</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#10b981;border:2px solid #fff"></span> GT Team 1</div>
      </div>
    </div>
  </div>
</div>

<div class="footer">Generated by TacticalVision AI — VM_FOOTBALL</div>

<script>
const DATA = {{DATA_JSON}};

// KPIs
const players = DATA.player_metrics;
const totalDist = players.reduce((s,p)=>s+p.distance_m,0).toFixed(1);
const maxSpd = Math.max(...players.map(p=>p.max_speed_kmh)).toFixed(1);
const totalSprints = players.reduce((s,p)=>s+p.sprint_count,0);
document.getElementById('kpi-dist').textContent = totalDist;
document.getElementById('kpi-maxsp').textContent = maxSpd;
document.getElementById('kpi-sprints').textContent = totalSprints;

if(DATA.comparison_summary){
  const cs = DATA.comparison_summary;
  document.getElementById('kpi-rmse').textContent = cs.overall_rmse_m !== undefined ? cs.overall_rmse_m : '--';
  document.getElementById('kpi-cov').textContent = cs.coverage !== undefined ? (cs.coverage*100).toFixed(1) : '--';
  document.getElementById('kpi-prec').textContent = cs.precision !== undefined ? (cs.precision*100).toFixed(1) : '--';
}

// Team mapping legend
const tm = DATA.team_mapping || {};
document.getElementById('map0').textContent = tm['0'] !== undefined ? tm['0'] : '?';
document.getElementById('map1').textContent = tm['1'] !== undefined ? tm['1'] : '?';

// Player table
const tbody = document.querySelector('#playerTable tbody');
players.sort((a,b)=>b.distance_m - a.distance_m).forEach(p=>{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${p.track_id}</td><td>${p.team_id}</td><td>${p.role}</td><td>${p.minutes_visible.toFixed(2)}</td><td>${p.distance_m.toFixed(1)}</td><td>${p.max_speed_kmh.toFixed(1)}</td><td>${p.mean_speed_kmh.toFixed(1)}</td><td>${p.sprint_count}</td><td>${p.sprint_distance_m.toFixed(1)}</td><td>${p.high_speed_distance_m.toFixed(1)}</td>`;
  tbody.appendChild(tr);
});

// Charts
Chart.defaults.color='#94a3b8';
Chart.defaults.borderColor='#1e293b';

// Width
const widthCtx = document.getElementById('widthChart').getContext('2d');
const team0w = DATA.team_shape.filter(s=>s.team_id===0);
const team1w = DATA.team_shape.filter(s=>s.team_id===1);
new Chart(widthCtx, {
  type:'line',
  data:{
    labels: team0w.map(s=>s.time_s.toFixed(1)),
    datasets:[
      {label:'Team 0', data:team0w.map(s=>s.width_m), borderColor:'#3b82f6', tension:0.3, pointRadius:0},
      {label:'Team 1', data:team1w.map(s=>s.width_m), borderColor:'#10b981', tension:0.3, pointRadius:0}
    ]
  },
  options:{responsive:true, maintainAspectRatio:false, interaction:{mode:'index', intersect:false}, scales:{x:{title:{display:true,text:'Time (s)'}}, y:{title:{display:true,text:'Width (m)'}}}}
});

// Compactness
const compCtx = document.getElementById('compactChart').getContext('2d');
const team0c = DATA.team_shape.filter(s=>s.team_id===0 && s.compactness!==null);
const team1c = DATA.team_shape.filter(s=>s.team_id===1 && s.compactness!==null);
new Chart(compCtx, {
  type:'line',
  data:{
    labels: team0c.map(s=>s.time_s.toFixed(1)),
    datasets:[
      {label:'Team 0', data:team0c.map(s=>s.compactness), borderColor:'#3b82f6', tension:0.3, pointRadius:0},
      {label:'Team 1', data:team1c.map(s=>s.compactness), borderColor:'#10b981', tension:0.3, pointRadius:0}
    ]
  },
  options:{responsive:true, maintainAspectRatio:false, interaction:{mode:'index', intersect:false}, scales:{x:{title:{display:true,text:'Time (s)'}}, y:{title:{display:true,text:'Compactness'}}}}
});

// Error
const errCtx = document.getElementById('errorChart').getContext('2d');
const errData = DATA.frame_errors.filter(e=>e.mean_error_m!==null);
new Chart(errCtx, {
  type:'line',
  data:{
    labels: errData.map(e=>((e.frame_id-1)/25.0).toFixed(1)),
    datasets:[
      {label:'Mean Error (m)', data:errData.map(e=>e.mean_error_m), borderColor:'#f59e0b', backgroundColor:'rgba(245,158,11,0.1)', fill:true, tension:0.3, pointRadius:0}
    ]
  },
  options:{responsive:true, maintainAspectRatio:false, interaction:{mode:'index', intersect:false}, scales:{x:{title:{display:true,text:'Time (s)'}}, y:{title:{display:true,text:'Error (m)'}}}}
});

// Speed profiles
const spdCtx = document.getElementById('speedChart').getContext('2d');
const spdDatasets = Object.entries(DATA.speed_profiles).map(([tid, pts], idx)=>{
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6'];
  return {label:`Player ${tid}`, data:pts.map(p=>p.speed_kmh), borderColor:colors[idx%colors.length], tension:0.3, pointRadius:0};
});
if(spdDatasets.length>0){
  const spdLabels = Object.values(DATA.speed_profiles)[0].map(p=>p.time_s.toFixed(1));
  new Chart(spdCtx, {
    type:'line',
    data:{labels:spdLabels, datasets:spdDatasets},
    options:{responsive:true, maintainAspectRatio:false, interaction:{mode:'index', intersect:false}, scales:{x:{title:{display:true,text:'Time (s)'}}, y:{title:{display:true,text:'Speed (km/h)'}}}}
  });
}

// 2D Pitch
const canvas = document.getElementById('pitchCanvas');
const ctx = canvas.getContext('2d');
const W=105, H=68;
const margin=20;
const scl = (canvas.width - margin*2) / W;
const sclY = (canvas.height - margin*2) / H;

function toCanvas(x,y){ return [margin + x*scl, margin + y*sclY]; }

function drawPitch(){
  ctx.fillStyle='#1a3c1a';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='rgba(255,255,255,0.6)';
  ctx.lineWidth=2;
  const [x0,y0]=toCanvas(0,0);
  const [x1,y1]=toCanvas(W,H);
  ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  const [xm,_]=toCanvas(52.5,0);
  ctx.beginPath(); ctx.moveTo(xm,y0); ctx.lineTo(xm,y1); ctx.stroke();
  const [cx,cy]=toCanvas(52.5,34);
  ctx.beginPath(); ctx.arc(cx,cy,9.15*scl,0,Math.PI*2); ctx.stroke();
}

function drawFrame(fid){
  drawPitch();
  const frameData = DATA.positions[String(fid)];
  if(!frameData) return;
  const teamMap = DATA.team_mapping || {};
  // GT squares
  frameData.gt.forEach(p=>{
    const [x,y]=toCanvas(p.x,p.y);
    ctx.strokeStyle = p.team_id===0 ? '#3b82f6' : '#10b981';
    ctx.lineWidth=2;
    ctx.strokeRect(x-4,y-4,8,8);
  });
  // Pred circles (mapped team color)
  frameData.pred.forEach(p=>{
    const mapped = teamMap[String(p.team_id)] !== undefined ? teamMap[String(p.team_id)] : p.team_id;
    const [x,y]=toCanvas(p.x,p.y);
    ctx.fillStyle = mapped===0 ? '#3b82f6' : '#10b981';
    ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill();
  });
}

const slider = document.getElementById('frameSlider');
const frameLabel = document.getElementById('frameLabel');
const timeLabel = document.getElementById('timeLabel');

slider.addEventListener('input', e=>{
  const fid = parseInt(e.target.value);
  frameLabel.textContent = fid;
  timeLabel.textContent = ((fid-1)/25.0).toFixed(2);
  drawFrame(fid);
});

drawFrame(1);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--output_html", type=str, required=True)
    args = parser.parse_args()

    base_dir = os.path.dirname(args.output_html)
    os.makedirs(base_dir, exist_ok=True)

    data = build_dashboard_data(args.sequence_name, os.path.dirname(base_dir))

    duration_s = data["metadata"]["duration_s"]
    html = HTML_TEMPLATE
    html = html.replace("{{SEQ_NAME}}", args.sequence_name)
    html = html.replace("{{DURATION}}", f"{duration_s:.1f}s")
    html = html.replace("{{FRAMES}}", str(data["metadata"]["frames"]))
    html = html.replace("{{FPS}}", str(data["metadata"]["fps"]))
    html = html.replace("{{DATA_JSON}}", json.dumps(data))

    with open(args.output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(args.output_html) / 1024.0
    print(f"Dashboard generated: {args.output_html} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
