"""Custom visualisations rendered as self-contained HTML (canvas / SVG) inside Streamlit components.

Each function takes real, precomputed data and returns HTML. Nothing here computes model output.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import streamlit.components.v1 as components

from ML.integration_engineer.control_tower.theme import (
    AMBER, CLASS_COLOR, CLASS_LABEL, FONT_LINK, INK, LIME, MUTED, PAPER, PLATE, PLATE_2, RED, RULE, SOFT,
)

BASE = f"""<link rel="stylesheet" href="{FONT_LINK}"><style>
html,body{{margin:0;padding:0;background:{INK};color:{PAPER};font-family:'IBM Plex Mono',monospace;overflow:hidden}}
svg{{display:block;width:100%;height:auto}} text{{font-family:'IBM Plex Mono',monospace;fill:{MUTED}}}
.disp{{font-family:'Big Shoulders Display',sans-serif}}
#tip{{position:fixed;display:none;pointer-events:none;background:{PLATE_2};border:1px solid {RULE};color:{PAPER};font:400 11px/1.45 'IBM Plex Mono',monospace;padding:6px 8px;max-width:270px;z-index:9;white-space:pre-line}}
</style><div id="tip"></div>"""

TIP_JS = """<script>(function(){const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const n=e.target.closest&&e.target.closest('[data-tip]');
if(!n){tip.style.display='none';return}tip.textContent=n.getAttribute('data-tip');tip.style.display='block';
const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(e.clientX+12,innerWidth-w-6)+'px';tip.style.top=Math.max(4,Math.min(e.clientY+12,innerHeight-h-6))+'px';});})();</script>"""


def show(html: str, height: int) -> None:
    components.html(html, height=height, scrolling=False)


def show_svg(svg: str, height: int) -> None:
    show(BASE + svg + TIP_JS, height)


# --------------------------------------------------------------------------------------------
# Globe (d3-geo orthographic, canvas)
# --------------------------------------------------------------------------------------------
def globe_html(ports: list, kpi: dict, height: int = 560) -> str:
    return BASE + """
<div id="g" style="position:relative;width:100%;height:__H__px;background:#0A0A0A;cursor:grab">
<canvas id="c" style="position:absolute;inset:0;width:100%;height:100%" role="img" aria-label="Rotatable globe with Terminal Petikemas Surabaya marked and PortWatch reference ports in grey"></canvas>
<div style="position:absolute;left:0;bottom:0;font-size:10.5px;line-height:1.7;color:#8E8E86;background:rgba(10,10,10,.82);padding:6px 8px 4px 0">
<div><span style="display:inline-block;width:9px;height:9px;background:#C7F36B;margin-right:6px"></span><b style="color:#F4F4EE;font-weight:500">Observed terminal</b> · has event data</div>
<div><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#77776f;margin:0 8px 0 2px"></span>Reference port · size = container vessels, PortWatch directory · <b style="color:#F2A93B;font-weight:500">no congestion data</b></div>
<div style="opacity:.8">Drag to rotate · scroll to zoom</div></div>
<div style="position:absolute;right:0;top:0;display:flex;gap:1px">
<button id="z1" style="all:unset;cursor:pointer;background:#1A1A18;color:#BDBDB4;padding:5px 9px;font-size:10.5px">World</button>
<button id="z2" style="all:unset;cursor:pointer;background:#C7F36B;color:#0A0A0A;padding:5px 9px;font-size:10.5px">Surabaya</button></div>
<div id="err" style="display:none;position:absolute;inset:0;align-items:center;justify-content:center;color:#8E8E86;font-size:12px;text-align:center;padding:2rem">The map library could not load. Surabaya, Indonesia: 7°12′S 112°43′E.</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"></script>
<script>
const PORTS=__PORTS__, HOME=[112.7217,-7.2009], KPI=__KPI__;
const root=document.getElementById('g'),canvas=document.getElementById('c'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
if(!window.d3||!window.topojson){document.getElementById('err').style.display='flex';canvas.style.display='none';}
else(async function(){
let land;try{const w=await (await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json')).json();land=topojson.feature(w,w.objects.land);}catch(e){document.getElementById('err').style.display='flex';return;}
const proj=d3.geoOrthographic().clipAngle(90).precision(.4),path=d3.geoPath(proj,ctx),grat=d3.geoGraticule().step([15,15])();
let W,H,base,k=1.0,rot=[-HOME[0]+18,-HOME[1]+4];
const rmax=d3.max(PORTS,p=>p[4]),rs=d3.scaleSqrt().domain([1,rmax]).range([.7,5.6]);
function size(){const d=devicePixelRatio||1;W=root.clientWidth;H=root.clientHeight;canvas.width=W*d;canvas.height=H*d;ctx.setTransform(d,0,0,d,0,0);base=Math.min(W,H)*.47;draw();}
function draw(){proj.scale(base*k).translate([W/2,H/2]).rotate(rot);ctx.clearRect(0,0,W,H);
ctx.beginPath();path({type:'Sphere'});ctx.fillStyle='#0D0D0C';ctx.fill();ctx.strokeStyle='#2B2B27';ctx.lineWidth=1;ctx.stroke();
ctx.beginPath();path(grat);ctx.strokeStyle='#191917';ctx.lineWidth=.7;ctx.stroke();
ctx.beginPath();path(land);ctx.fillStyle='#1E1E1B';ctx.fill();ctx.strokeStyle='#3b3b35';ctx.lineWidth=.5;ctx.stroke();
const c=proj.rotate(),center=[-c[0],-c[1]];
for(const p of PORTS){if(d3.geoDistance([p[3],p[2]],center)>Math.PI/2)continue;const xy=proj([p[3],p[2]]);ctx.beginPath();ctx.arc(xy[0],xy[1],rs(Math.max(p[4],1))*Math.sqrt(k)*.9,0,6.3);ctx.fillStyle='rgba(150,150,140,.55)';ctx.fill();}
if(d3.geoDistance(HOME,center)<Math.PI/2){const h=proj(HOME);ctx.strokeStyle='#C7F36B';ctx.lineWidth=1;
ctx.beginPath();ctx.arc(h[0],h[1],15,0,6.3);ctx.stroke();ctx.beginPath();ctx.arc(h[0],h[1],27,0,6.3);ctx.globalAlpha=.35;ctx.stroke();ctx.globalAlpha=1;
ctx.fillStyle='#C7F36B';ctx.fillRect(h[0]-4,h[1]-4,8,8);
const lx=Math.min(h[0]+32,W-208),ly=Math.max(h[1]-58,34);ctx.strokeStyle='#C7F36B';ctx.globalAlpha=.6;ctx.beginPath();ctx.moveTo(h[0]+11,h[1]-11);ctx.lineTo(lx,ly+40);ctx.stroke();ctx.globalAlpha=1;
ctx.fillStyle='#0A0A0A';ctx.fillRect(lx,ly-22,200,66);
ctx.fillStyle='#F4F4EE';ctx.font="700 19px 'Big Shoulders Stencil Display',sans-serif";ctx.fillText('TPS SURABAYA',lx+6,ly-3);
ctx.fillStyle='#BDBDB4';ctx.font="400 10.5px 'IBM Plex Mono',monospace";ctx.fillText('7°12′S 112°43′E · ID SUB',lx+6,ly+12);ctx.fillText(KPI.n+' stays · Jul–Sep 2021',lx+6,ly+26);ctx.fillStyle='#FF5A47';ctx.fillText(KPI.crit+' critical delay',lx+6,ly+39);}
}
function fly(toRot,toK,ms){if(reduce){rot=toRot;k=toK;draw();return}const r0=rot.slice(),k0=k;d3.select({}).transition().duration(ms).ease(d3.easeCubicInOut).tween('z',()=>t=>{rot=[r0[0]+(toRot[0]-r0[0])*t,r0[1]+(toRot[1]-r0[1])*t];k=k0+(toK-k0)*t;draw();});}
d3.select(canvas).call(d3.drag().on('drag',e=>{const s=70/(base*k/ 2.2);rot=[rot[0]+e.dx*s,Math.max(-85,Math.min(85,rot[1]-e.dy*s))];draw();}));
canvas.addEventListener('wheel',e=>{e.preventDefault();k=Math.max(.8,Math.min(30,k*Math.exp(-e.deltaY*.0014)));draw();},{passive:false});
canvas.addEventListener('mousemove',e=>{const b=canvas.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top,c=proj.rotate(),center=[-c[0],-c[1]];let best=null,bd=9;
const h=proj(HOME);if(Math.hypot(h[0]-mx,h[1]-my)<14&&d3.geoDistance(HOME,center)<Math.PI/2){tip.textContent='Terminal Petikemas Surabaya\\nObserved terminal: real event log\\n'+KPI.n+' stays, Jul–Sep 2021\\nHistorical, not live';}
else{for(const p of PORTS){if(d3.geoDistance([p[3],p[2]],center)>Math.PI/2)continue;const q=proj([p[3],p[2]]),d=Math.hypot(q[0]-mx,q[1]-my);if(d<bd){bd=d;best=p}}
if(!best){tip.style.display='none';return}tip.textContent=best[0]+', '+best[1]+'\\nReference only: '+best[4].toLocaleString()+' container vessels in PortWatch directory\\nNo congestion or dwell data';}
tip.style.display='block';tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-6)+'px';tip.style.top=Math.min(e.clientY+12,innerHeight-tip.offsetHeight-6)+'px';});
canvas.addEventListener('mouseleave',()=>tip.style.display='none');
document.getElementById('z1').onclick=()=>fly([-HOME[0]+18,-HOME[1]+4],1,900);
document.getElementById('z2').onclick=()=>fly([-HOME[0],-HOME[1]],14,1100);
addEventListener('resize',size);size();if(document.fonts&&document.fonts.ready)document.fonts.ready.then(draw);
})();
</script>""".replace("__H__", str(height)).replace("__PORTS__", json.dumps(ports, separators=(",", ":"))).replace("__KPI__", json.dumps(kpi))


# --------------------------------------------------------------------------------------------
# Terminal Replay (signature): the real event log as a time-lapse of stacked containers
# --------------------------------------------------------------------------------------------
def replay_html(replay: dict, height: int = 590) -> str:
    return BASE + """
<style>
#w{position:relative;width:100%;height:__H__px;background:#0A0A0A}
#hud{position:absolute;left:0;top:0;right:0;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none}
#clock{font:800 46px/1 'Big Shoulders Stencil Display',sans-serif;color:#F4F4EE;letter-spacing:.01em}
#ctx{font:400 11px/1.4 'IBM Plex Mono',monospace;color:#F2A93B;margin-top:3px;min-height:15px}
#leg{font:400 10.5px/1.7 'IBM Plex Mono',monospace;color:#8E8E86;text-align:right}
#leg i{display:inline-block;width:12px;height:7px;margin:0 5px 0 12px}
#ctl{position:absolute;left:0;right:0;bottom:0;display:flex;gap:10px;align-items:center;height:30px}
#ctl button{all:unset;cursor:pointer;background:#C7F36B;color:#0A0A0A;font:600 11px 'IBM Plex Mono',monospace;padding:6px 10px;min-width:44px;text-align:center}
#ctl button.sec{background:#1A1A18;color:#BDBDB4;font-weight:400} #ctl button:focus-visible{outline:2px solid #F4F4EE;outline-offset:2px}
#sc{flex:1;accent-color:#C7F36B;height:18px} #note{font:400 10.5px 'IBM Plex Mono',monospace;color:#8E8E86;white-space:nowrap}
@media(max-width:760px){#clock{font-size:32px}#leg i{margin-left:6px}#note{display:none}}
</style>
<div id="w"><canvas id="c" style="position:absolute;inset:0;width:100%;height:calc(100% - 34px)" role="img" aria-label="Animated replay of container stays moving through customs, delivery order and truck pickup"></canvas>
<div id="hud"><div><div id="clock"></div><div id="ctx"></div></div>
<div id="leg"><div>Colour = time already spent in the yard</div><div><i style="background:#C7F36B"></i>up to 65 h<i style="background:#F2A93B"></i>65–161 h<i style="background:#FF5A47"></i>over 161 h</div><div>Short box 20 ft · long box 40 ft · 1 box = __EVERY__ real containers</div></div></div>
<div id="ctl"><button id="pp" aria-label="Play or pause">Pause</button><button class="sec" id="sp" aria-label="Change speed">1×</button><input id="sc" type="range" min="0" max="1000" value="0" aria-label="Replay time"><span id="note"></span></div></div>
<script>
const R=__DATA__,S=R.stays,HOURS=R.hours,P50=R.p50,P85=R.p85,EVERY=R.sample_every,START=new Date(R.start+'T00:00:00');
const STN=[{t:'Waiting for customs',s:'Discharged; customs release not yet known'},{t:'Waiting for delivery order',s:'Customs released; no delivery job yet'},{t:'Waiting for a truck',s:'Delivery job created; truck not at the gate'}];
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),sc=document.getElementById('sc'),ppb=document.getElementById('pp'),spb=document.getElementById('sp'),clockEl=document.getElementById('clock'),ctxEl=document.getElementById('ctx'),noteEl=document.getElementById('note');
const boxes=S.map((s,i)=>({d:s[0],c:s[1],j:s[2],ti:s[3],o:s[4],l:s[5],x:-20,y:0,on:false}));
let W,H,hour=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,speed=1,last=0,lastDrawHour=-1,gone=0;
const SPEEDS=[1,2,4];function size(){const d=devicePixelRatio||1,b=cv.getBoundingClientRect();W=b.width;H=b.height;cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0);}
function station(b,h){if(h<b.d||h>=b.ti)return -1;if(h<b.c)return 0;if(h<b.j)return 1;return 2;}
function col(a){return a<=P50?'#C7F36B':a<=P85?'#F2A93B':'#FF5A47';}
function frame(ts){if(!last)last=ts;const dt=Math.min(ts-last,60);last=ts;
if(playing){hour+=dt/1000*14*speed;if(hour>=HOURS){hour=0;boxes.forEach(b=>{b.on=false})}sc.value=hour/HOURS*1000;}
const gap=Math.max(14,W*.02),sw=(W-gap*2)/3,floor=H-34,top=92,unit=Math.max(4,Math.min(11,sw/42)),bh=Math.max(3,unit*.62),rowH=bh+1.5;
const groups=[[],[],[]];let yard=[0,0,0];let out=0;
for(const b of boxes){const st=station(b,hour);if(st<0){if(hour>=b.ti)out++;b.on=false;b.st=-1;continue}groups[st].push(b);yard[st]++;b.st=st;}
ctx.clearRect(0,0,W,H);
for(let i=0;i<3;i++){const x0=i*(sw+gap);ctx.strokeStyle='#2B2B27';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x0,floor+.5);ctx.lineTo(x0+sw,floor+.5);ctx.stroke();
ctx.fillStyle='#F4F4EE';ctx.font="700 19px 'Big Shoulders Display',sans-serif";ctx.fillText(STN[i].t.toUpperCase(),x0,floor+19);
const g=groups[i],over=g.filter(b=>hour-b.d>P85).length;
ctx.textAlign='right';ctx.font="500 13px 'IBM Plex Mono',monospace";ctx.fillStyle='#F4F4EE';ctx.fillText((g.length*EVERY).toLocaleString(),x0+sw,floor+19);ctx.textAlign='left';
ctx.font="400 10.5px 'IBM Plex Mono',monospace";ctx.fillStyle='#8E8E86';if(W>760)ctx.fillText(STN[i].s,x0,floor+32);
if(over>0){ctx.textAlign='right';ctx.fillStyle='#FF5A47';ctx.fillText((over*EVERY).toLocaleString()+' over 161 h',x0+sw,floor+32);ctx.textAlign='left'}
// Oldest at the bottom: long-waiting containers are buried under the new arrivals.
g.sort((a,b)=>a.d-b.d);let cx=x0,row=0;
for(const b of g){const w=(b.l?2:1)*unit;if(cx+w>x0+sw){cx=x0;row++}const tx=cx,ty=floor-(row+1)*rowH;cx+=w+1;
if(!b.on){b.x=i===0&&b.c>b.d&&hour-b.d<3?x0-6:tx;b.y=b.c<=b.d&&i===1?ty-14:ty-30;if(i>0)b.x=x0+sw*.5;b.on=true}
const ease=Math.min(1,dt*.012);b.x+=(tx-b.x)*ease;b.y+=(ty-b.y)*ease;ctx.fillStyle=col(hour-b.d);ctx.fillRect(b.x,Math.max(top,b.y),w,bh);}
}
const date=new Date(START.getTime()+hour*3600000),key=date.toISOString().slice(0,10);
const local=new Date(START.getTime()+hour*3600000);const iso=local.getFullYear()+'-'+String(local.getMonth()+1).padStart(2,'0')+'-'+String(local.getDate()).padStart(2,'0');
clockEl.textContent=local.toLocaleDateString('en-GB',{day:'2-digit',month:'short'}).toUpperCase()+' '+String(local.getHours()).padStart(2,'0')+':00';
ctxEl.textContent=R.notes[iso]||'';
noteEl.textContent=((yard[0]+yard[1]+yard[2])*EVERY).toLocaleString()+' in the yard · real 2021 events, not live';
requestAnimationFrame(frame);}
sc.addEventListener('input',()=>{hour=sc.value/1000*HOURS;boxes.forEach(b=>b.on=false);});
ppb.onclick=()=>{playing=!playing;ppb.textContent=playing?'Pause':'Play'};ppb.textContent=playing?'Pause':'Play';
spb.onclick=()=>{speed=SPEEDS[(SPEEDS.indexOf(speed)+1)%3];spb.textContent=speed+'×'};
addEventListener('resize',size);size();requestAnimationFrame(frame);
</script>""".replace("__H__", str(height)).replace("__EVERY__", str(replay["sample_every"])).replace("__DATA__", json.dumps(replay, separators=(",", ":")))


# --------------------------------------------------------------------------------------------
# Ridgeline
# --------------------------------------------------------------------------------------------
def _fmt_week(iso: str) -> str:
    import datetime as dt
    return dt.date.fromisoformat(iso).strftime("%d %b").upper()


def ridgeline_svg(ov: dict) -> str:
    W, row, amp, left, right = 880, 30, 74, 74, 16
    ridges, axis = ov["ridges"], ov["ridge_axis"]
    H = 110 + row * (len(ridges) - 1) + 26
    lx = lambda h: left + (math.log10(h) - math.log10(4)) / (math.log10(800) - math.log10(4)) * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Ridgeline of dwell time density by discharge week">']
    for h, lab, c in [(65.11, "65 h", AMBER), (160.9, "161 h", RED)]:
        out.append(f'<line x1="{lx(h):.1f}" x2="{lx(h):.1f}" y1="10" y2="{H-22}" stroke="{c}" stroke-width="1" stroke-dasharray="2 4"/><text x="{lx(h)+4:.1f}" y="19" font-size="10.5" style="fill:{c}">{lab}</text>')
    peak = max(max(r["density"]) for r in ridges)
    for i, r in enumerate(ridges):
        base = 96 + i * row
        pts = [(lx(a), base - d / peak * amp) for a, d in zip(axis, r["density"])]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        hot = r["crit"] >= 0.2
        stroke = RED if hot else PAPER
        out.append(f'<g data-tip="Week of {_fmt_week(r["wk"])}\n{r["n"]:,} discharges\nMedian dwell {r["median"]:.0f} h\n{r["crit"]*100:.1f}% critical delay">'
                   f'<polygon points="{lx(axis[0]):.1f},{base} {path} {lx(axis[-1]):.1f},{base}" fill="{INK}"/>'
                   f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="1.4"/>'
                   f'<rect x="0" y="{base-row}" width="{W}" height="{row}" fill="transparent"/></g>'
                   f'<text x="0" y="{base-1}" font-size="10.5" style="fill:{RED if hot else MUTED}">{_fmt_week(r["wk"])}</text>')
    base = 96 + (len(ridges) - 1) * row
    for h in (6, 12, 24, 48, 96, 192, 384, 768):
        out.append(f'<text x="{lx(h):.1f}" y="{base+18}" font-size="10" text-anchor="middle">{h} h</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Radial terminal clock
# --------------------------------------------------------------------------------------------
def clock_svg(matrix: List[List[int]], title: str) -> str:
    S, cx, cy, r0, ring = 460, 230, 230, 62, 23
    names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    peak = max(max(r) for r in matrix)
    out = [f'<svg viewBox="0 0 {S} {S}" style="max-width:490px;margin:0 auto" role="img" aria-label="{title} by weekday and hour">']
    for d, row in enumerate(matrix):
        inner, outer = r0 + d * ring, r0 + (d + 1) * ring - 2
        for h, v in enumerate(row):
            a0, a1 = (h / 24) * 2 * math.pi - math.pi / 2 + .012, ((h + 1) / 24) * 2 * math.pi - math.pi / 2 - .012
            pts = [(cx + outer * math.cos(a0), cy + outer * math.sin(a0)), (cx + outer * math.cos(a1), cy + outer * math.sin(a1)),
                   (cx + inner * math.cos(a1), cy + inner * math.sin(a1)), (cx + inner * math.cos(a0), cy + inner * math.sin(a0))]
            alpha = (v / peak) ** .75
            out.append(f'<path d="M{pts[0][0]:.1f} {pts[0][1]:.1f}A{outer} {outer} 0 0 1 {pts[1][0]:.1f} {pts[1][1]:.1f}L{pts[2][0]:.1f} {pts[2][1]:.1f}A{inner} {inner} 0 0 0 {pts[3][0]:.1f} {pts[3][1]:.1f}Z" '
                       f'fill="{LIME}" fill-opacity="{0.04 + alpha * .96:.3f}" data-tip="{full[d]} {h:02d}:00\n{v:,} containers"/>')
        out.append(f'<text x="{cx+3}" y="{cy-inner-6}" font-size="9" style="fill:{SOFT}">{names[d]}</text>')
    for h in (0, 6, 12, 18):
        a = h / 24 * 2 * math.pi - math.pi / 2
        out.append(f'<text x="{cx + (r0 + 7*ring + 9) * math.cos(a):.1f}" y="{cy + (r0 + 7*ring + 9) * math.sin(a) + 4:.1f}" font-size="11" text-anchor="middle" style="fill:{PAPER}">{h:02d}</text>')
    out.append(f'<text x="{cx}" y="{cy+2}" font-size="17" text-anchor="middle" class="disp" style="fill:{PAPER};font-weight:700">{title.upper()}</text>')
    out.append(f'<text x="{cx}" y="{cy+16}" font-size="9.5" text-anchor="middle">local hour</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Calendar: critical-delay rate by discharge day
# --------------------------------------------------------------------------------------------
def calendar_svg(days: List[dict]) -> str:
    import datetime as dt
    cell, gap, left, top = 40, 3, 44, 32
    first = dt.date.fromisoformat(days[0]["d"])
    start = first - dt.timedelta(days=first.weekday())
    rows = {}
    for d in days:
        x = dt.date.fromisoformat(d["d"])
        rows[((x - start).days // 7, x.weekday())] = (x, d)
    nweek = max(k[0] for k in rows) + 1
    W, H = left + 7 * (cell + gap) + 150, top + nweek * (cell * .62 + gap) + 28
    out = [f'<svg viewBox="0 0 {W} {H:.0f}" role="img" aria-label="Calendar of daily critical-delay rate">']
    for i, n in enumerate("MON TUE WED THU FRI SAT SUN".split()):
        out.append(f'<text x="{left + i*(cell+gap) + cell/2}" y="{top-10}" font-size="10" text-anchor="middle">{n}</text>')
    prev_label = None
    for (w, dow), (x, d) in sorted(rows.items()):
        y = top + w * (cell * .62 + gap)
        xx = left + dow * (cell + gap)
        crit = d["crit"] or 0
        small = d["n"] < 40
        alpha = 0 if small else min(1, crit / .45) ** .8
        fill = f'fill="{RED}" fill-opacity="{0.06 + alpha*.94:.3f}"' if not small else f'fill="{PLATE_2}"'
        extra = f'\n{d["holiday"]}' if d["holiday"] else ""
        tip = f'{x:%a %d %b}\n{d["n"]:,} discharges\n{crit*100:.1f}% later critical\nMedian dwell {d["median"]:.0f} h\n{d["ppkm"]}{extra}'
        out.append(f'<rect x="{xx}" y="{y:.1f}" width="{cell}" height="{cell*.62:.1f}" {fill} data-tip="{tip}"/>')
        out.append(f'<text x="{xx+3}" y="{y+10:.1f}" font-size="8.5" style="fill:{PAPER};opacity:.6;pointer-events:none">{x.day}</text>')
        if d["holiday"]:
            out.append(f'<rect x="{xx}" y="{y + cell*.62 - 3:.1f}" width="{cell}" height="3" fill="{PAPER}" style="pointer-events:none"/>')
        if dow == 0 or (x, d) == list(sorted(rows.items()))[0][1]:
            label = f"{x:%b}".upper()
            if label != prev_label:
                out.append(f'<text x="0" y="{y+14:.1f}" font-size="10.5" style="fill:{SOFT}">{label}</text>')
                prev_label = label
    lx = left + 7 * (cell + gap) + 14
    out.append(f'<text x="{lx}" y="{top+6}" font-size="10">Later critical</text>')
    for i, v in enumerate((0, .1, .2, .3, .45)):
        a = min(1, v / .45) ** .8
        out.append(f'<rect x="{lx}" y="{top + 14 + i*16}" width="20" height="13" fill="{RED}" fill-opacity="{0.06 + a*.94:.3f}"/><text x="{lx+26}" y="{top + 25 + i*16}" font-size="10">{int(v*100)}%</text>')
    out.append(f'<rect x="{lx}" y="{top + 102}" width="20" height="3" fill="{PAPER}"/><text x="{lx+26}" y="{top + 107}" font-size="10">Public holiday</text>')
    out.append(f'<rect x="{lx}" y="{top + 116}" width="20" height="13" fill="{PLATE_2}"/><text x="{lx+26}" y="{top + 127}" font-size="10">Under 40 stays</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Decision triangles (probability simplex)
# --------------------------------------------------------------------------------------------
def _simplex_xy(p0: float, p1: float, p2: float, S: float, pad: float):
    h = S * math.sqrt(3) / 2
    # Corners: ON_TIME bottom-left, AT_RISK top, CRITICAL bottom-right
    x = pad + (p2 + p1 / 2) * S
    y = pad + h - p1 * h
    return x, y


def simplex_group(cells: dict, color: str, S: float, pad: float, marker: Optional[List[float]] = None) -> str:
    n = cells["n_bins"] if "n_bins" in cells else 22
    out = []
    data = cells["cells"]
    if data:
        peak = max(c[2] for c in data)
        total = sum(c[2] for c in data)
    for i, j, c in data:
        p0, p1 = (i + .5) / n, (j + .5) / n
        p2 = 1 - p0 - p1
        if p2 < -0.03:
            continue
        p2 = max(p2, 0)
        x, y = _simplex_xy(p0, p1, p2, S, pad)
        a = (c / peak) ** .55
        r = S / n * .62
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{color}" fill-opacity="{.1 + a*.9:.3f}" data-tip="{c:,} stays ({c/total*100:.1f}%)\nP(on time) ≈ {p0:.2f}\nP(at risk) ≈ {p1:.2f}\nP(critical) ≈ {p2:.2f}"/>')
    return "".join(out)


def simplex_frame(S: float, pad: float, corner: bool = True) -> str:
    a, b, c = _simplex_xy(1, 0, 0, S, pad), _simplex_xy(0, 1, 0, S, pad), _simplex_xy(0, 0, 1, S, pad)
    centre = _simplex_xy(1/3, 1/3, 1/3, S, pad)
    out = [f'<polygon points="{a[0]},{a[1]} {b[0]},{b[1]} {c[0]},{c[1]}" fill="none" stroke="{RULE}"/>']
    for corner_pt in (a, b, c):
        out.append(f'<line x1="{centre[0]:.1f}" y1="{centre[1]:.1f}" x2="{(corner_pt[0]+centre[0])/2:.1f}" y2="{(corner_pt[1]+centre[1])/2:.1f}" stroke="{RULE}" stroke-dasharray="2 3"/>')
    if corner:
        out.append(f'<text x="{a[0]-2}" y="{a[1]+15}" font-size="10.5" style="fill:{SOFT}">On time</text>'
                   f'<text x="{b[0]}" y="{b[1]-6}" font-size="10.5" text-anchor="middle" style="fill:{SOFT}">At risk</text>'
                   f'<text x="{c[0]+2}" y="{c[1]+15}" font-size="10.5" text-anchor="end" style="fill:{SOFT}">Critical</text>')
    return "".join(out)


def simplex_triptych(simplex: dict) -> str:
    S, pad, W = 256, 24, 960
    h = S * math.sqrt(3) / 2
    H = h + pad * 2 + 22
    out = [f'<svg viewBox="0 0 {W} {H:.0f}" style="max-width:1000px;margin:0 auto" role="img" aria-label="Where the model places stays of each true class">']
    for idx, name in enumerate(("ON_TIME", "AT_RISK", "CRITICAL_DELAY")):
        block = simplex["by_class"][name]
        data = {"n_bins": simplex["n_bins"], "cells": block["cells"]}
        out.append(f'<g transform="translate({idx * (W/3):.1f},18)"><text x="{pad}" y="-4" font-size="11" style="fill:{PAPER}">Truly {CLASS_LABEL[name].lower()} · {block["n"]:,} stays</text>'
                   + simplex_frame(S, pad) + simplex_group(data, CLASS_COLOR[name], S, pad) + "</g>")
    return "".join(out) + "</svg>"


def simplex_single(simplex: dict, proba: List[float], name: str) -> str:
    S, pad = 236, 24
    h = S * math.sqrt(3) / 2
    W, H = S + pad * 2, h + pad * 2 + 16
    out = [f'<svg viewBox="0 0 {W} {H:.0f}" role="img" aria-label="This container\'s position among validation predictions">', simplex_frame(S, pad)]
    for cls in ("ON_TIME", "AT_RISK", "CRITICAL_DELAY"):
        block = simplex["by_class"][cls]
        out.append(simplex_group({"n_bins": simplex["n_bins"], "cells": block["cells"]}, MUTED, S, pad).replace('fill-opacity="', 'fill-opacity="0.35" data-o="'))
    x, y = _simplex_xy(proba[0], proba[1], proba[2], S, pad)
    c = CLASS_COLOR[name]
    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" stroke="{c}"/><circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="{INK}" stroke-width="2" data-tip="This container\nOn time {proba[0]*100:.1f}%\nAt risk {proba[1]*100:.1f}%\nCritical {proba[2]*100:.1f}%"/>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Information staircase
# --------------------------------------------------------------------------------------------
def staircase_svg(horizons: dict, oracles: List[dict]) -> str:
    W, H, L, B, T = 960, 300, 42, 62, 18
    steps = [(v["label"], v["mean_fold_macro_f1"], False, f'{v["n"]:,} open stays · CI {v["ci95"][0]:.3f}–{v["ci95"][1]:.3f}') for v in horizons.values()]
    steps += [(o["label"], o["macro_f1"], True, "Uses future information") for o in oracles]
    n = len(steps)
    sw = (W - L) / n
    y = lambda v: T + (1 - v) * (H - T - B) / 1
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Macro-F1 rising as more information becomes available">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" x2="{W}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#1c1c1a"/><text x="{L-8}" y="{y(v)+3:.1f}" font-size="10" text-anchor="end">{v:.2f}</text>')
    out.append(f'<line x1="{L}" x2="{W}" y1="{y(.9):.1f}" y2="{y(.9):.1f}" stroke="{SOFT}" stroke-dasharray="4 4"/><text x="{L+4}" y="{y(.9)-5:.1f}" font-size="10" style="fill:{SOFT}">Research target 0.90</text>')
    for i, (lab, v, ghost, note) in enumerate(steps):
        x = L + i * sw
        if ghost:
            out.append(f'<rect x="{x+1:.1f}" y="{y(v):.1f}" width="{sw-2:.1f}" height="{y(0)-y(v):.1f}" fill="none" stroke="{AMBER}" stroke-dasharray="3 3" data-tip="{lab}\nMacro-F1 {v:.3f}\n{note}"/>')
        else:
            out.append(f'<rect x="{x+1:.1f}" y="{y(v):.1f}" width="{sw-2:.1f}" height="{y(0)-y(v):.1f}" fill="{LIME}" data-tip="{lab}\nMacro-F1 {v:.3f}\n{note}"/>')
        out.append(f'<text x="{x+sw/2:.1f}" y="{y(v)-7:.1f}" text-anchor="middle" font-size="19" class="disp" style="fill:{AMBER if ghost else PAPER};font-weight:700">{v:.3f}</text>')
        words, line = lab.split(" "), ""
        lines = []
        for w_ in words:
            if len(line) + len(w_) > 17:
                lines.append(line)
                line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for li, t in enumerate(lines[:3]):
            out.append(f'<text x="{x+sw/2:.1f}" y="{y(0)+15+li*12:.1f}" text-anchor="middle" font-size="10" style="fill:{AMBER if ghost else SOFT}">{t}</text>')
    x_sep = L + 4 * sw
    out.append(f'<line x1="{x_sep:.1f}" x2="{x_sep:.1f}" y1="{T}" y2="{H-14}" stroke="{MUTED}"/>')
    out.append(f'<text x="{x_sep-6:.1f}" y="{T+8}" text-anchor="end" font-size="10.5" style="fill:{LIME}">Real models</text><text x="{x_sep+6:.1f}" y="{T+8}" font-size="10.5" style="fill:{AMBER}">Bounds, future information</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Confusion mosaic: column widths and cell heights proportional to counts
# --------------------------------------------------------------------------------------------
def confusion_svg(cm: List[List[int]]) -> str:
    W, H, L, T = 560, 330, 86, 28
    names = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
    totals = [sum(r) for r in cm]
    grand = sum(totals)
    PW, PH = W - L - 6, H - T - 8
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Confusion mosaic; row height is the true class share, cell width the predicted share">']
    y = T
    out.append(f'<text x="{L}" y="12" font-size="10">Predicted share within each true class →</text>')
    for i, row in enumerate(cm):
        rh = totals[i] / grand * PH
        x = L
        out.append(f'<text x="0" y="{y+rh/2+3:.1f}" font-size="10.5" style="fill:{SOFT}">{CLASS_LABEL[names[i]]}</text>')
        for j, v in enumerate(row):
            w = v / totals[i] * PW
            correct = i == j
            c = CLASS_COLOR[names[j]]
            op = .95 if correct else .28
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w-1.5,0):.1f}" height="{rh-1.5:.1f}" fill="{c}" fill-opacity="{op}" data-tip="True: {CLASS_LABEL[names[i]]}\nPredicted: {CLASS_LABEL[names[j]]}\n{v:,} stays · {v/totals[i]*100:.1f}% of this true class"/>')
            if w > 44 and rh > 22:
                out.append(f'<text x="{x+6:.1f}" y="{y+rh/2+4:.1f}" font-size="11.5" style="fill:{INK if correct else PAPER};font-weight:500;pointer-events:none">{v/totals[i]*100:.0f}%</text>')
            x += w
        y += rh
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Feature rank bump chart
# --------------------------------------------------------------------------------------------
FAMILY_COLOR = {"Customs": LIME, "Delivery & gate": AMBER, "Quarantine & inspection": RED, "Vessel call": PAPER, "Terminal state": SOFT, "Container & time": MUTED}


def bump_svg(bump: dict, labels: Dict[str, str]) -> str:
    H_KEYS = ["A", "B12", "B24", "B48"]
    names = {"A": "At discharge", "B12": "+12 h", "B24": "+24 h", "B48": "+48 h"}
    W, H, L, R, T, depth = 960, 372, 170, 170, 30, 12
    xs = {h: L + i * (W - L - R) / 3 for i, h in enumerate(H_KEYS)}
    y = lambda r: T + (r - 1) * (H - T - 14) / (depth - 1)
    rank = {h: {f: i + 1 for i, f in enumerate(bump["by_horizon"][h])} for h in H_KEYS}
    share = bump["by_horizon"]
    feats = bump["top"]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rank of the top features at each prediction horizon">']
    for h in H_KEYS:
        out.append(f'<line x1="{xs[h]:.1f}" x2="{xs[h]:.1f}" y1="{T-8}" y2="{H-6}" stroke="#1c1c1a"/><text x="{xs[h]:.1f}" y="14" font-size="11" text-anchor="middle" style="fill:{PAPER}">{names[h]}</text>')
    for f in sorted(feats, key=lambda z: rank["B48"].get(z, 99), reverse=True):
        c = FAMILY_COLOR[bump["family"].get(f, "Container & time")]
        pts = [(xs[h], y(rank[h][f])) for h in H_KEYS if rank[h].get(f, 99) <= depth]
        if not pts:
            continue
        lab = labels.get(f, f)
        tip = lab + "\n" + bump["family"].get(f, "") + "\n" + "\n".join(f'{names[h]}: rank {rank[h].get(f, "–")} · {share[h].get(f, 0)*100:.1f}% of gain' for h in H_KEYS)
        d = "M" + " L".join(f"{x:.1f} {yy:.1f}" for x, yy in pts)
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/><path d="{d}" fill="none" stroke="transparent" stroke-width="12" data-tip="{tip}"/>')
        for x, yy in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="4.5" fill="{c}" stroke="{INK}" stroke-width="2" data-tip="{tip}"/>')
        out.append(f'<text x="{pts[0][0]-9:.1f}" y="{pts[0][1]+3.5:.1f}" font-size="10.5" text-anchor="end" style="fill:{SOFT}">{lab}</text>')
        out.append(f'<text x="{pts[-1][0]+9:.1f}" y="{pts[-1][1]+3.5:.1f}" font-size="10.5" style="fill:{SOFT}">{lab}</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------
# Yard bay plan (schematic heat grid)
# --------------------------------------------------------------------------------------------
def bay_svg(blocks: List[dict], weeks: List[str]) -> str:
    cell_w, cell_h, L, T, R = 58, 19, 46, 26, 150
    blocks = sorted(blocks, key=lambda b: -b["crit"])
    W, H = L + cell_w * len(weeks) + R, T + cell_h * len(blocks) + 8
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Critical-delay rate by yard block and discharge week">']
    for j, w in enumerate(weeks):
        out.append(f'<text x="{L + j*cell_w + 2}" y="{T-8}" font-size="9.5">{_fmt_week(w)}</text>')
    for i, b in enumerate(blocks):
        y = T + i * cell_h
        out.append(f'<text x="0" y="{y+13}" font-size="11" style="fill:{PAPER}">{b["block"]}</text>')
        for j, w in enumerate(weeks):
            c = b["weeks"].get(w)
            x = L + j * cell_w
            if not c:
                out.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{PLATE}"/>')
                continue
            a = min(1, c["crit"] / .5) ** .8
            out.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{RED}" fill-opacity="{.05 + a*.95:.3f}" data-tip="Block {b["block"]}, week of {_fmt_week(w)}\n{c["n"]:,} discharges\n{c["crit"]*100:.1f}% critical delay"/>')
        out.append(f'<text x="{L + len(weeks)*cell_w + 8}" y="{y+13}" font-size="10">{b["n"]:,} · {b["crit"]*100:.0f}% crit · {b["median"]:.0f} h</text>')
    return "".join(out) + "</svg>"
