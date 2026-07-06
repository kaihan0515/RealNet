# -*- coding: utf-8 -*-
"""
RealNet 訓練儀表板（本機網頁介面）
- 設定訓練參數（類別 / batch size / epoch / 早停 patience...）並一鍵啟動、停止訓練
- 即時顯示:訓練損失曲線、驗證 AUROC 曲線、最佳權重、早停倒數、GPU 使用率、日誌
- 只用 Python 標準庫 + PyYAML,無需額外安裝

啟動:  python train_dashboard.py  (或直接執行 train_dashboard.bat)
之後瀏覽器開 http://127.0.0.1:8123
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
PORT = 8123

DATASET_DIRS = {
    "MVTec-AD": ("mvtec", "sdas"),
    "VisA": ("visa", "sdas"),
    "MPDD": ("mpdd", "sdas"),
    "BTAD": ("btad", "sdas"),
}

CATEGORIES = {
    "MVTec-AD": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                 "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
    "VisA": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
             "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
    "MPDD": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"],
    "BTAD": ["01", "02", "03"],
}

STATE = {
    "proc": None,
    "started_at": None,
    "params": None,
    "run_log": None,
    "exit_code": None,
}
LOCK = threading.Lock()

LOSS_RE = re.compile(r"Epoch: \[(\d+)/(\d+)\].*?Iter: \[(\d+)/(\d+)\].*?Loss ([\d.eE+-]+) \(([\d.eE+-]+)\)")


def build_config(params):
    """以 experiments/{dataset}/realnet.yaml 為底,套上介面參數,寫出 realnet_ui.yaml"""
    dataset = params["dataset"]
    base_path = os.path.join(ROOT, "experiments", dataset, "realnet.yaml")
    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["batch_size"] = int(params["batch_size"])
    cfg["dataset"]["workers"] = int(params["workers"])
    cfg["trainer"]["max_epoch"] = int(params["max_epoch"])
    cfg["trainer"]["val_freq_epoch"] = int(params["val_freq_epoch"])
    cfg["trainer"]["optimizer"]["kwargs"]["lr"] = float(params["lr"])
    cfg["trainer"]["early_stop"] = {
        "enabled": bool(params["early_stop_enabled"]),
        "patience": int(params["patience"]),
        "min_delta": float(params["min_delta"]),
    }
    out_path = os.path.join(ROOT, "experiments", dataset, "realnet_ui.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def start_training(params):
    with LOCK:
        if STATE["proc"] is not None and STATE["proc"].poll() is None:
            return False, "訓練已在進行中"
        build_config(params)
        os.makedirs(os.path.join(ROOT, "ui_runs"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_log = os.path.join(ROOT, "ui_runs", "run_{}_{}.log".format(params["class_name"], stamp))
        cmd = [
            PYTHON, "train_realnet.py",
            "--dataset", params["dataset"],
            "--class_name", params["class_name"],
            "--config", "experiments/{}/realnet_ui.yaml",
        ]
        logf = open(run_log, "w", encoding="utf-8", errors="replace")
        logf.write("cmd: {}\n\n".format(" ".join(cmd)))
        logf.flush()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        STATE.update(proc=proc, started_at=time.time(), params=dict(params),
                     run_log=run_log, exit_code=None)
        return True, "已啟動 (pid={})".format(proc.pid)


def stop_training():
    with LOCK:
        proc = STATE["proc"]
        if proc is None or proc.poll() is not None:
            return False, "目前沒有進行中的訓練"
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            proc.terminate()
        return True, "已送出停止指令"


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4).stdout.strip()
        name, util, used, total, temp = [x.strip() for x in out.split(",")[:5]]
        return {"name": name, "util": int(util), "mem_used": int(used),
                "mem_total": int(total), "temp": int(temp)}
    except Exception:
        return None


def latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def read_history(dataset, class_name):
    log_dir = os.path.join(ROOT, "experiments", dataset, "realnet_log")
    path = latest_file(os.path.join(log_dir, "realnet_{}_*_history.csv".format(class_name)))
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def read_summary(dataset, class_name):
    log_dir = os.path.join(ROOT, "experiments", dataset, "realnet_log")
    path = latest_file(os.path.join(log_dir, "realnet_{}_*_summary.json".format(class_name)))
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        s["_file"] = os.path.basename(path)
        return s
    except Exception:
        return None


def parse_run_log(run_log, max_points=400):
    """從執行日誌抓損失曲線與最後幾行"""
    loss_pts, tail = [], []
    if not run_log or not os.path.exists(run_log):
        return loss_pts, tail, None
    try:
        with open(run_log, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return loss_pts, tail, None
    cur_epoch = None
    for ln in lines:
        m = LOSS_RE.search(ln)
        if m:
            cur_epoch = (int(m.group(1)), int(m.group(2)))
            loss_pts.append({"iter": int(m.group(3)), "loss": float(m.group(6))})
    if len(loss_pts) > max_points:
        step = len(loss_pts) / max_points
        loss_pts = [loss_pts[int(i * step)] for i in range(max_points)]
    tail = [ln.rstrip() for ln in lines[-25:]]
    return loss_pts, tail, cur_epoch


def dataset_ready(dataset):
    img_dir, sdas_dir = DATASET_DIRS[dataset]
    base = os.path.join(ROOT, "data", dataset)
    return {
        "images": os.path.isdir(os.path.join(base, img_dir)) and bool(os.listdir(os.path.join(base, img_dir))) if os.path.isdir(os.path.join(base, img_dir)) else False,
        "sdas": os.path.isdir(os.path.join(base, sdas_dir)) and bool(os.listdir(os.path.join(base, sdas_dir))) if os.path.isdir(os.path.join(base, sdas_dir)) else False,
    }


def checkpoint_info(dataset, class_name):
    path = os.path.join(ROOT, "experiments", dataset, "realnet_checkpoints", class_name, "ckpt_best.pth.tar")
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return {"path": os.path.relpath(path, ROOT), "size_mb": round(st.st_size / 1e6, 1),
            "mtime": time.strftime("%H:%M:%S", time.localtime(st.st_mtime))}


def get_status(dataset, class_name):
    with LOCK:
        proc = STATE["proc"]
        running = proc is not None and proc.poll() is None
        if proc is not None and not running and STATE["exit_code"] is None:
            STATE["exit_code"] = proc.poll()
        params = STATE["params"]
        run_log = STATE["run_log"]
        started = STATE["started_at"]
        exit_code = STATE["exit_code"]

    if params:  # 執行中以實際訓練的 dataset/class 為準
        dataset = params["dataset"]
        class_name = params["class_name"]

    loss_pts, tail, cur_epoch = parse_run_log(run_log)
    history = read_history(dataset, class_name)
    summary = read_summary(dataset, class_name)

    return {
        "running": running,
        "exit_code": exit_code,
        "elapsed": int(time.time() - started) if started and running else
                   (int((os.path.getmtime(run_log) if run_log and os.path.exists(run_log) else started) - started) if started else None),
        "params": params,
        "dataset": dataset,
        "class_name": class_name,
        "cur_epoch": cur_epoch,
        "loss": loss_pts,
        "history": history,
        "summary": summary,
        "gpu": gpu_info(),
        "log_tail": tail,
        "data_ready": dataset_ready(dataset),
        "checkpoint": checkpoint_info(dataset, class_name),
    }


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>RealNet 訓練儀表板</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1420;--card:#171e2e;--line:#26304a;--txt:#e8ecf5;--sub:#8b96ad;
--acc:#4f8cff;--ok:#3ecf8e;--warn:#f5a623;--bad:#ff5d5d;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.6 "Segoe UI","Microsoft JhengHei",sans-serif;padding:18px}
h1{font-size:20px;margin-bottom:4px}
.sub{color:var(--sub);font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:320px 1fr;gap:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:14px;color:var(--sub);margin-bottom:10px;font-weight:600}
label{display:block;font-size:12px;color:var(--sub);margin:8px 0 2px}
input,select{width:100%;background:#0d1322;border:1px solid var(--line);color:var(--txt);
border-radius:8px;padding:7px 9px;font-size:13px}
input:focus,select:focus{outline:none;border-color:var(--acc)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.btns{display:flex;gap:8px;margin-top:14px}
button{flex:1;border:none;border-radius:9px;padding:10px;font-size:14px;font-weight:700;cursor:pointer}
#btnStart{background:var(--acc);color:#fff}#btnStart:disabled{background:#2a3956;color:#67718a;cursor:not-allowed}
#btnStop{background:#33202a;color:var(--bad);border:1px solid #5a2b38}#btnStop:disabled{opacity:.4;cursor:not-allowed}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:13px;font-weight:700}
.b-run{background:#12301f;color:var(--ok)}.b-idle{background:#252d40;color:var(--sub)}
.b-done{background:#153050;color:var(--acc)}.b-err{background:#3a1a1a;color:var(--bad)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:14px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px}
.stat .k{font-size:11px;color:var(--sub)}.stat .v{font-size:19px;font-weight:800;margin-top:2px}
.stat .v small{font-size:11px;color:var(--sub);font-weight:400}
.bar{height:7px;background:#0d1322;border-radius:4px;margin-top:6px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);border-radius:4px;transition:width .5s}
canvas{width:100%;height:190px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
@media(max-width:1100px){.charts{grid-template-columns:1fr}}
#log{background:#0a0f1a;border-radius:8px;padding:10px;font:11px/1.5 Consolas,monospace;
height:200px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#9fb0cc}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.kv{font-size:12px;color:var(--sub)}.kv b{color:var(--txt)}
#summaryCard{display:none;border-color:#2b4a2b}
table{width:100%;font-size:12px;border-collapse:collapse}
td{padding:3px 6px;border-bottom:1px solid var(--line);color:var(--sub)}td:last-child{text-align:right;color:var(--txt);font-weight:600}
</style></head><body>
<h1>RealNet 訓練儀表板 <span id="stBadge" class="badge b-idle">待命</span></h1>
<div class="sub">kaihan0515/RealNet · 早停版訓練 · <span id="clock"></span></div>

<div class="stats">
 <div class="stat"><div class="k">訓練進度 (epoch)</div><div class="v" id="epochV">—</div><div class="bar"><i id="epochBar" style="width:0%"></i></div></div>
 <div class="stat"><div class="k">最新 Loss</div><div class="v" id="lossV">—</div></div>
 <div class="stat"><div class="k">目前指標 (image+pixel AUROC 均值)</div><div class="v" id="curM">—</div></div>
 <div class="stat"><div class="k">最佳指標 <small id="bestEp"></small></div><div class="v ok" id="bestM">—</div></div>
 <div class="stat"><div class="k">早停倒數 (無提升驗證次數)</div><div class="v" id="esV">—</div><div class="bar"><i id="esBar" style="width:0%;background:var(--warn)"></i></div></div>
 <div class="stat"><div class="k">GPU <small id="gpuName"></small></div><div class="v" id="gpuV">—</div><div class="bar"><i id="gpuBar" style="width:0%;background:var(--ok)"></i></div></div>
 <div class="stat"><div class="k">已耗時</div><div class="v" id="elapsed">—</div></div>
</div>

<div class="grid">
 <div>
  <div class="card">
   <h2>訓練參數</h2>
   <label>資料集</label><select id="dataset"></select>
   <label>類別</label><select id="class_name"></select>
   <div class="row2">
    <div><label>Batch size</label><input id="batch_size" type="number" value="8"></div>
    <div><label>DataLoader workers</label><input id="workers" type="number" value="4"></div>
   </div>
   <div class="row2">
    <div><label>最大 Epoch</label><input id="max_epoch" type="number" value="1000"></div>
    <div><label>驗證頻率 (epoch)</label><input id="val_freq_epoch" type="number" value="5"></div>
   </div>
   <label>學習率</label><input id="lr" type="number" step="0.00001" value="0.0001">
   <div class="row2">
    <div><label>早停 patience (驗證次數)</label><input id="patience" type="number" value="10"></div>
    <div><label>min_delta</label><input id="min_delta" type="number" step="0.0001" value="0.0001"></div>
   </div>
   <label style="display:flex;align-items:center;gap:6px;margin-top:10px">
     <input id="early_stop_enabled" type="checkbox" checked style="width:auto"> 啟用早停
   </label>
   <div class="btns">
    <button id="btnStart" onclick="startTrain()">▶ 開始訓練</button>
    <button id="btnStop" onclick="stopTrain()" disabled>■ 停止</button>
   </div>
   <div id="msg" class="kv" style="margin-top:8px"></div>
  </div>
  <div class="card" style="margin-top:14px">
   <h2>資料集 / 權重狀態</h2>
   <table>
    <tr><td>原始影像資料</td><td id="dsImg">—</td></tr>
    <tr><td>SDAS 合成異常</td><td id="dsSdas">—</td></tr>
    <tr><td>最佳權重檔</td><td id="ckpt">尚未產生</td></tr>
   </table>
  </div>
  <div class="card" id="summaryCard" style="margin-top:14px">
   <h2 class="ok">✔ 訓練總結 (summary json)</h2>
   <table id="summaryTbl"></table>
  </div>
 </div>

 <div>
  <div class="charts">
   <div class="card"><h2>訓練損失</h2><canvas id="cLoss" width="640" height="190"></canvas></div>
   <div class="card"><h2>驗證 AUROC(每次驗證,虛線=最佳)</h2><canvas id="cMet" width="640" height="190"></canvas></div>
  </div>
  <div class="card"><h2>訓練日誌 (最後 25 行)</h2><div id="log">尚無日誌</div></div>
 </div>
</div>

<script>
const CATS = __CATS__;
const dsSel = document.getElementById('dataset'), clsSel = document.getElementById('class_name');
Object.keys(CATS).forEach(d=>dsSel.add(new Option(d,d)));
function fillCls(){clsSel.innerHTML='';CATS[dsSel.value].forEach(c=>clsSel.add(new Option(c,c)));}
dsSel.onchange=fillCls; fillCls();

function fmtT(s){if(s==null)return '—';const h=~~(s/3600),m=~~(s%3600/60);return h?`${h}h ${m}m`:`${m}m ${s%60}s`;}
async function startTrain(){
  const p={dataset:dsSel.value,class_name:clsSel.value};
  ['batch_size','workers','max_epoch','val_freq_epoch','lr','patience','min_delta'].forEach(k=>p[k]=document.getElementById(k).value);
  p.early_stop_enabled=document.getElementById('early_stop_enabled').checked;
  const r=await fetch('/api/start',{method:'POST',body:JSON.stringify(p)}).then(r=>r.json());
  document.getElementById('msg').textContent=r.msg;
}
async function stopTrain(){
  if(!confirm('確定停止訓練?已存的最佳權重不會遺失。'))return;
  const r=await fetch('/api/stop',{method:'POST'}).then(r=>r.json());
  document.getElementById('msg').textContent=r.msg;
}

function drawLine(cv,pts,ys,opt){
  const c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  if(!pts.length){c.fillStyle='#5a6480';c.font='12px sans-serif';c.fillText('等待資料…',W/2-30,H/2);return;}
  const pad=34, xs=pts.map((_,i)=>i);
  let lo=Math.min(...ys), hi=Math.max(...ys); if(hi-lo<1e-6){hi=lo+1e-6}
  const X=i=>pad+(W-pad-8)*(xs.length<2?0:i/(xs.length-1));
  const Y=v=>H-18-(H-30)*(v-lo)/(hi-lo);
  c.strokeStyle='#26304a';c.beginPath();c.moveTo(pad,8);c.lineTo(pad,H-18);c.lineTo(W-6,H-18);c.stroke();
  c.fillStyle='#8b96ad';c.font='10px Consolas';
  c.fillText(hi.toFixed(opt.dec),2,14);c.fillText(lo.toFixed(opt.dec),2,H-20);
  c.strokeStyle=opt.color;c.lineWidth=1.6;c.beginPath();
  ys.forEach((v,i)=>i?c.lineTo(X(i),Y(v)):c.moveTo(X(i),Y(v)));c.stroke();
  if(opt.best!=null){c.setLineDash([4,4]);c.strokeStyle='#3ecf8e';c.beginPath();
    c.moveTo(pad,Y(opt.best));c.lineTo(W-6,Y(opt.best));c.stroke();c.setLineDash([]);}
  c.lineWidth=1;
}

let wasRunning=false;
async function tick(){
  let s; try{s=await fetch('/api/status?dataset='+dsSel.value+'&class_name='+clsSel.value).then(r=>r.json());}catch(e){return;}
  const badge=document.getElementById('stBadge');
  if(s.running){badge.className='badge b-run';badge.textContent='訓練中';}
  else if(s.summary && s.summary.early_stopped){badge.className='badge b-done';badge.textContent='已完成(早停)';}
  else if(s.exit_code===0){badge.className='badge b-done';badge.textContent='已完成';}
  else if(s.exit_code!=null){badge.className='badge b-err';badge.textContent='異常結束(code '+s.exit_code+')';}
  else{badge.className='badge b-idle';badge.textContent='待命';}
  document.getElementById('btnStart').disabled=s.running;
  document.getElementById('btnStop').disabled=!s.running;

  if(s.cur_epoch){const[e,me]=s.cur_epoch;
    document.getElementById('epochV').innerHTML=e+' <small>/ '+me+'</small>';
    document.getElementById('epochBar').style.width=(100*e/me)+'%';}
  document.getElementById('lossV').textContent=s.loss.length?s.loss[s.loss.length-1].loss.toFixed(4):'—';
  document.getElementById('elapsed').textContent=fmtT(s.elapsed);

  const h=s.history;
  if(h.length){
    const last=h[h.length-1];
    document.getElementById('curM').textContent=(+last.key_metric).toFixed(4);
    document.getElementById('bestM').textContent=(+last.best_metric).toFixed(4);
    document.getElementById('bestEp').textContent='@ epoch '+last.best_epoch;
    const pat=s.params?+s.params.patience:10;
    document.getElementById('esV').innerHTML=last.no_improve_rounds+' <small>/ '+pat+'</small>';
    document.getElementById('esBar').style.width=(100*last.no_improve_rounds/pat)+'%';
    drawLine(document.getElementById('cMet'),h,h.map(r=>+r.key_metric),{color:'#4f8cff',dec:3,best:+last.best_metric});
  }
  drawLine(document.getElementById('cLoss'),s.loss,s.loss.map(p=>p.loss),{color:'#f5a623',dec:3});

  if(s.gpu){document.getElementById('gpuName').textContent=s.gpu.name;
    document.getElementById('gpuV').innerHTML=s.gpu.util+'% <small>'+s.gpu.mem_used+'/'+s.gpu.mem_total+'MB · '+s.gpu.temp+'°C</small>';
    document.getElementById('gpuBar').style.width=(100*s.gpu.mem_used/s.gpu.mem_total)+'%';}

  document.getElementById('dsImg').innerHTML=s.data_ready.images?'<span class="ok">✔ 就緒</span>':'<span class="bad">✘ 未就緒</span>';
  document.getElementById('dsSdas').innerHTML=s.data_ready.sdas?'<span class="ok">✔ 就緒</span>':'<span class="bad">✘ 未就緒</span>';
  document.getElementById('ckpt').innerHTML=s.checkpoint?('<span class="ok">'+s.checkpoint.path+'</span> ('+s.checkpoint.size_mb+'MB, '+s.checkpoint.mtime+')'):'尚未產生';

  if(s.log_tail.length){const el=document.getElementById('log');
    el.textContent=s.log_tail.join('\n');el.scrollTop=el.scrollHeight;}

  const sc=document.getElementById('summaryCard');
  if(s.summary){sc.style.display='block';
    const rows={'類別':s.summary.class_name,'最佳指標':(+s.summary.best_metric).toFixed(5),
      '最佳 epoch':s.summary.best_epoch,'停止 epoch':s.summary.stopped_epoch,
      '是否早停':s.summary.early_stopped?'是':'否','patience':s.summary.early_stop_patience,
      '紀錄檔':s.summary._file};
    document.getElementById('summaryTbl').innerHTML=Object.entries(rows).map(([k,v])=>'<tr><td>'+k+'</td><td>'+v+'</td></tr>').join('');
  } else sc.style.display='none';
  wasRunning=s.running;
}
setInterval(tick,2000); tick();
setInterval(()=>document.getElementById('clock').textContent=new Date().toLocaleTimeString(),1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 靜音 http 存取日誌
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            dataset = q.get("dataset", ["MVTec-AD"])[0]
            class_name = q.get("class_name", ["bottle"])[0]
            try:
                self._json(get_status(dataset, class_name))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif self.path == "/" or self.path.startswith("/index"):
            body = PAGE.replace("__CATS__", json.dumps(CATEGORIES)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            n = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(n) or b"{}")
            ok, msg = start_training(params)
            self._json({"ok": ok, "msg": msg})
        elif self.path == "/api/stop":
            ok, msg = stop_training()
            self._json({"ok": ok, "msg": msg})
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:{}".format(PORT)
    print("RealNet 訓練儀表板: {}".format(url))
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n關閉儀表板")


if __name__ == "__main__":
    main()
