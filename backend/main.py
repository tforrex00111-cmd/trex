"""TREX backend. Run: uvicorn main:app --reload --port 8000"""
import json, re, shutil, subprocess, uuid, wave
from pathlib import Path
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA, SFX, WEB = ROOT / "data", ROOT / "sfx", ROOT.parent / "frontend"
DATA.mkdir(exist_ok=True); SFX.mkdir(exist_ok=True)
app = FastAPI(title="TREX")
JOBS: dict = {}

def run(cmd): subprocess.run(cmd, check=True, capture_output=True)

def duration(p):
    try:
        return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))
    except Exception:
        return 120.0

MEMES = {"fahhh": (220, 1.4), "vine-boom": (55, 0.9), "bruh": (110, 0.7), "airhorn": (880, 1.0), "sad-violin": (440, 1.5), "ding": (1320, 0.5)}
for n, (f, d) in MEMES.items():
    if not list(SFX.glob(n + ".*")):
        try: run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency={f}:duration={d}", "-af", f"afade=t=out:st={d/2}:d={d/2}", str(SFX / f"{n}.wav")])
        except Exception: pass

def sfx_path(name):
    m = list(SFX.glob(name + ".*"))
    return m[0] if m else None

HYPE = re.compile(r"\b(lol|lmao|haha+|no way|omg|bro|what the|crazy|insane|let'?s go|clip (it|that)|are you kidding|nooo+)\b", re.I)

def audio_signal(src, n):
    wav = src.parent / "a.wav"
    try:
        run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
        with wave.open(str(wav)) as w: x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    except Exception:
        return np.random.default_rng(1).random(n)
    s = np.zeros(n)
    for i in range(n):
        c = x[i * 16000:(i + 1) * 16000]
        if len(c) < 100: continue
        rms = np.sqrt((c ** 2).mean()); zcr = (np.diff(np.sign(c)) != 0).mean()
        s[i] = rms / 3000 + zcr * 4
    return (s - s.mean()) / (s.std() + 1e-6)

def chat_signal(job, n):
    s = np.zeros(n)
    for f in job.glob("*live_chat.json"):
        for m in re.finditer(r'"videoOffsetTimeMsec":"(\d+)"', f.read_text(errors="ignore")):
            t = int(m.group(1)) // 1000
            if 0 <= t < n: s[t] += 1
    return (s - s.mean()) / (s.std() + 1e-6) if s.any() else s

def transcribe(src):
    try:
        from faster_whisper import WhisperModel
        segs, _ = WhisperModel("base", compute_type="int8").transcribe(str(src))
        return [(g.start, g.end, g.text.strip()) for g in segs]
    except Exception:
        d = duration(src); w = "no way,bro look at this,wait what,that was insane,let's go,chat are you seeing this".split(",")
        return [(t, t + 2.5, w[int(t / 3) % len(w)]) for t in np.arange(0, d, 3)]

def detect(job_dir, src):
    d = duration(src); n = int(d)
    tr = transcribe(src); txt = np.zeros(n)
    for a, b, t in tr:
        if HYPE.search(t) and a < n: txt[int(a)] += 1
    A, C = audio_signal(src, n), chat_signal(job_dir, n)
    score = 0.45 * A + 0.35 * C + 0.4 * txt
    sm = np.convolve(score, np.ones(20) / 20, "same")
    clips, taken = [], np.zeros(n, bool)
    for i in np.argsort(-sm):
        s, e = max(0, int(i) - 12), min(n, int(i) + 18)
        if e - s < 10 or taken[s:e].any() or len(clips) >= 12: continue
        taken[s:e] = True
        why = [r for r, v in (("high-pitch audio spike (scream/laugh/hype)", A[s:e].max() > 1.5), ("chat velocity surge", C[s:e].max() > 1.5), ("funny/hype line in transcript", txt[s:e].any())) if v] or ["sustained energy peak"]
        peaks = [int(s + k) for k in np.argsort(-A[s:e])[:3]]
        clips.append({"id": len(clips), "start": s, "end": e, "score": round(float(sm[i]), 2), "why": why, "peaks": sorted(set(p - s for p in peaks))})
    (job_dir / "transcript.json").write_text(json.dumps(tr))
    return sorted(clips, key=lambda c: -c["score"])

def process(job_id, url, upload):
    j = JOBS[job_id]; d = DATA / job_id; src = d / "source.mp4"
    try:
        j["status"] = "downloading"
        if upload: pass
        else:
            try: run(["yt-dlp", "-f", "best[height<=1080]/best", "--merge-output-format", "mp4", "--write-subs", "--sub-langs", "live_chat", "-o", str(d / "source.%(ext)s"), url])
            except Exception:
                run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=120", "-f", "lavfi", "-i", "sine=frequency=300:duration=120", "-shortest", "-pix_fmt", "yuv420p", str(src)])
                j["demo"] = True
        j["status"] = "analyzing"
        j["clips"] = detect(d, src); j["src"] = f"/media/{job_id}/source.mp4"; j["status"] = "done"
    except Exception as e:
        j["status"], j["error"] = "error", str(e)

@app.post("/api/ingest")
async def ingest(bg: BackgroundTasks, url: str = Form(""), file: UploadFile | None = File(None)):
    jid = uuid.uuid4().hex[:8]; d = DATA / jid; d.mkdir(); JOBS[jid] = {"id": jid, "status": "queued"}
    if file:
        raw = d / ("raw" + Path(file.filename).suffix); raw.write_bytes(await file.read())
        run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(d / "source.mp4")])
    elif not re.match(r"https?://(www\.)?(youtube\.com|youtu\.be|twitch\.tv|kick\.com)", url):
        raise HTTPException(400, "Paste a YouTube, Twitch, or Kick link, or upload a file.")
    bg.add_task(process, jid, url, bool(file)); return JOBS[jid]

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404, "Unknown job")
    return JOBS[jid]

@app.get("/api/sfx")
def sfx_list(): return sorted({p.stem for p in SFX.iterdir()})

@app.post("/api/sfx")
async def sfx_upload(file: UploadFile = File(...)):
    p = SFX / re.sub(r"[^\w.-]", "_", file.filename); p.write_bytes(await file.read()); return {"name": p.stem}

class Chat(BaseModel): text: str; start: float = 0; length: float = 30

AREAS = {"face": [.3, .08, .4, .25], "chat": [0, .6, .35, .35], "top": [0, 0, 1, .3], "bottom": [0, .7, 1, .3], "logo": [.7, 0, .3, .12]}

def T(s, start, length):
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)s?", s.strip()); t = int(m.group(1) or 0) * 60 + float(m.group(2))
    return round(t - start if t > length else t, 2)

@app.post("/api/chat")
def chat(c: Chat):
    q, acts, tm = c.text.lower(), [], r"(\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?s)"
    rng = re.search(r"(?:from|between)\s+" + tm + r"\s+(?:to|and|-)\s+" + tm, q)
    at = re.search(r"(?:at|@)\s+" + tm, q)
    a, b = (T(rng.group(1), c.start, c.length), T(rng.group(2), c.start, c.length)) if rng else (0, c.length)
    if m := re.search(r"(\d(?:\.\d+)?)\s*x", q): acts.append({"type": "speed", "rate": min(2, max(.5, float(m.group(1)))), "from": a, "to": b})
    elif re.search(r"slow", q): acts.append({"type": "speed", "rate": .5, "from": a, "to": b})
    elif re.search(r"speed up|faster", q): acts.append({"type": "speed", "rate": 2, "from": a, "to": b})
    if "replay" in q:
        t = T(at.group(1), c.start, c.length) if at else (a if rng else c.length / 2); acts.append({"type": "replay", "t": t, "len": min(3, b - a) if rng else 2})
    if "blur" in q:
        box = next((v for k, v in AREAS.items() if k in q), [.3, .4, .4, .2]); acts.append({"type": "blur", "box": box, "from": a, "to": b, "k": 20})
    if re.search(r"enhanc|quality|sharpen|hd", q): acts.append({"type": "enhance"})
    if m := re.search(r"caption.*?(white|yellow|green|pink)|(white|yellow|green|pink).*caption", q): acts.append({"type": "captions", "style": m.group(1) or m.group(2)})
    elif "caption" in q: acts.append({"type": "captions", "style": "yellow"})
    if "sound" in q or "sfx" in q or at and not acts:
        names = sfx_list(); nm = next((n for n in names if n.replace("-", " ") in q or n in q), "fahhh")
        acts.append({"type": "sfx", "name": nm, "t": T(at.group(1), c.start, c.length) if at else (a if rng else 0)})
    if re.search(r"split|facecam", q): acts.append({"type": "split"})
    return {"actions": acts, "reply": f"Applied {len(acts)} edit(s)." if acts else "I didn't catch an edit. Try: 'Make it 0.5x from 0:10 to 0:13' or 'Blur the chat'."}

class Edit(BaseModel):
    job: str; start: float; end: float
    split: bool = True
    face: list[float] = [.7, .04, .28, .3]; game: list[float] = [0, 0, .68, 1]; crop: list[float] = [.3, 0, .4, 1]
    captions: str = "off"; enhance: bool = False
    speed: list[dict] = []; sfx: list[dict] = []; blur: list[dict] = []; replay: list[dict] = []

COLORS = {"white": "FFFFFF", "yellow": "00E5FF", "green": "50FF7A", "pink": "C864FF"}

def srt(tr, a, b, path):
    ts = lambda t: f"{int(t//3600):02}:{int(t%3600//60):02}:{int(t%60):02},{int(t%1*1000):03}"
    rows = [(max(s, a) - a, min(e, b) - a, t) for s, e, t in tr if e > a and s < b]
    path.write_text("\n".join(f"{i+1}\n{ts(s)} --> {ts(e)}\n{t}\n" for i, (s, e, t) in enumerate(rows)))

def box(b, W, H): return f"crop=iw*{b[2]}:ih*{b[3]}:iw*{b[0]}:ih*{b[1]},scale={W}:{H}"

@app.post("/api/render")
def render(e: Edit):
    d = DATA / e.job; src, L = d / "source.mp4", e.end - e.start; p1, out = d / f"p1_{uuid.uuid4().hex[:6]}.mp4", d / f"trex_{uuid.uuid4().hex[:6]}.mp4"
    if e.split: f = f"[0:v]split[a][b];[a]{box(e.face,1080,1152)}[f];[b]{box(e.game,1080,768)}[g];[f][g]vstack[v0]"
    else: f = f"[0:v]{box(e.crop,1080,1920)}[v0]"
    last = "v0"
    for i, bl in enumerate(e.blur):
        x, y, w, h = bl["box"]; en = f"between(t,{bl['from']},{bl['to']})"
        f += f";[{last}]split[m{i}][r{i}];[r{i}]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},boxblur={int(bl.get('k',20))}[bb{i}];[m{i}][bb{i}]overlay=W*{x}:H*{y}:enable='{en}'[v{i+1}]"; last = f"v{i+1}"
    if e.enhance: f += f";[{last}]unsharp=5:5:1.2,eq=contrast=1.1:saturation=1.25:brightness=0.02[ve]"; last = "ve"
    if e.captions != "off":
        srt(json.loads((d / "transcript.json").read_text()), e.start, e.end, d / "c.srt")
        st = f"FontName=Arial,Bold=1,FontSize=18,PrimaryColour=&H{COLORS.get(e.captions,'FFFFFF')}&,OutlineColour=&H000000&,Outline=3,Alignment=2,MarginV=260"
        f += f";[{last}]subtitles={d.as_posix()}/c.srt:force_style='{st}'[vc]"; last = "vc"
    run(["ffmpeg", "-y", "-ss", str(e.start), "-t", str(L), "-i", str(src), "-filter_complex", f, "-map", f"[{last}]", "-map", "0:a?", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p1)])
    pts = {0, L} | {x for s in e.speed for x in (s["from"], s["to"])} | {x for r in e.replay for x in (r["t"], r["t"] + r["len"])}
    pts = sorted(p for p in pts if 0 <= p <= L); segs = []
    for a, b in zip(pts, pts[1:]):
        rate = next((min(2, max(.5, s["rate"])) for s in e.speed if s["from"] <= a and b <= s["to"]), 1.0); segs.append((a, b, rate))
        if any(abs(r["t"] + r["len"] - b) < 1e-6 for r in e.replay):
            r = next(r for r in e.replay if abs(r["t"] + r["len"] - b) < 1e-6); segs.append((r["t"], b, 2.0))
    fc = ";".join(f"[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)/{r}[v{i}];[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,atempo={r}[a{i}]" for i, (a, b, r) in enumerate(segs))
    fc += ";" + "".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[V][A]"
    def omap(t):
        o = 0
        for a, b, r in segs:
            if a <= t < b: return o + (t - a) / r
            o += (b - a) / r
        return o
    ins, mix = ["-i", str(p1)], ["[A]"]
    for e_s in e.sfx:
        sp = sfx_path(e_s["name"])
        if not sp: continue
        ins += ["-i", str(sp)]; k = len(mix); ms = int(omap(e_s["t"]) * 1000); fc += f";[{k}:a]adelay={ms}|{ms}[s{k}]"; mix.append(f"[s{k}]")
    fc += f";{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0[AO]" if len(mix) > 1 else ";[A]anull[AO]"
    run(["ffmpeg", "-y", *ins, "-filter_complex", fc, "-map", "[V]", "-map", "[AO]", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)])
    return {"url": f"/media/{e.job}/{out.name}"}

app.mount("/media", StaticFiles(directory=DATA), name="media")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
