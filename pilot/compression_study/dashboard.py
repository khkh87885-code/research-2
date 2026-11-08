"""Self-contained local results workbench with no network dependencies."""
import json
from pathlib import Path

from .export import clean_rows, safe_metadata


def build_dashboard(rows, output_path, metadata=None, statistics=None):
    payload = {"rows": clean_rows(list(rows)), "metadata": safe_metadata(metadata)}
    # Escape HTML delimiters so identifiers cannot break out of the JSON element.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'">
<title>Local compression study results</title>
<style>body{font:15px system-ui,sans-serif;margin:0;background:#f5f7f9;color:#193045}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:26px}section{background:white;border:1px solid #dbe2e8;border-radius:10px;padding:18px;margin:16px 0}label{display:inline-block;margin:8px}select,button{padding:8px;border:1px solid #bbcbd7;border-radius:5px;background:white}button{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}.scroll{overflow:auto;max-height:520px}.metrics{display:flex;gap:20px;flex-wrap:wrap}.metric{padding:12px;background:#eef4f8;border-radius:7px}.bar{height:14px;background:#376a8c;display:inline-block;vertical-align:middle}.row{margin:8px 0}.note{color:#4c6274}svg{width:100%;height:330px}</style>
<main><h1>Local compression study results</h1><p class="note">Local file workbench. No patient images, network requests or external libraries. Filters apply to the records and CSV download.</p>
<section><label>Dataset <select id="dataset"></select></label><label>Split <select id="split"></select></label><label>Condition <select id="condition"></select></label><label>Prompt <select id="prompt_strategy"></select></label><label>Status <select id="status"></select></label><button id="reset">Reset filters</button> <button id="download">Download filtered CSV</button></section>
<section><div id="counts" class="metrics"></div><p class="note">Patients count unique dataset / patient pairs with known pseudonymous identity. Repeated image versions are not new patients. Chart means use successful finite Dice values only.</p></section>
<section><h2>Observed Dice by condition</h2><div id="chart"></div><p class="note">Descriptive image means for navigation; use exported patient statistics for inference. Mechanism and application prompts can be filtered separately.</p></section>
<section><h2>All matching records</h2><div class="scroll"><table id="records"></table></div></section><section><h2>Run metadata</h2><pre id="metadata"></pre></section>
<script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const payload=JSON.parse(document.getElementById('data').textContent), all=payload.rows;
const fields=['dataset','split','patient_key','image_key','condition','prompt_strategy','status','dice','iou','hd95','assd','saving_fraction','output_bytes','original_bytes','encode_ms','decode_ms','inference_ms'];
const keys=['dataset','split','condition','prompt_strategy','status'];
let current=[];
for(const key of keys){const el=document.getElementById(key);for(const value of ['',...new Set(all.map(r=>String(r[key]??'')))].sort()){const opt=document.createElement('option');opt.value=value;opt.textContent=value||'All';el.appendChild(opt);}el.addEventListener('change',render);}
function finite(x){return x!==null&&x!==''&&x!==undefined&&Number.isFinite(Number(x));}
function cell(tag,text){const el=document.createElement(tag);el.textContent=text??'';return el;}
function render(){current=all.filter(r=>keys.every(k=>!document.getElementById(k).value||String(r[k]??'')===document.getElementById(k).value));const ok=current.filter(r=>['ok','success','completed'].includes(String(r.status).toLowerCase()));const patients=new Set(current.filter(r=>r.patient_key).map(r=>JSON.stringify([r.dataset,r.patient_key])));const counts=document.getElementById('counts');counts.replaceChildren();for(const [label,value] of [['Records',current.length],['Successful',ok.length],['Failed/incomplete',current.length-ok.length],['Known patients',patients.size],['Missing patient keys',current.filter(r=>!r.patient_key).length]]){const el=cell('div',label+': '+value);el.className='metric';counts.appendChild(el);}
const table=document.getElementById('records');table.replaceChildren();const head=document.createElement('tr');fields.forEach(f=>head.appendChild(cell('th',f)));table.appendChild(head);for(const r of current){const tr=document.createElement('tr');fields.forEach(f=>tr.appendChild(cell('td',typeof r[f]==='number'?Number(r[f].toFixed(5)):r[f])));table.appendChild(tr);}
const chart=document.getElementById('chart');chart.replaceChildren();const groups=new Map();for(const r of ok){if(!finite(r.dice))continue;const label=[r.dataset,r.split,r.condition,r.prompt_strategy].join(' / ');if(!groups.has(label))groups.set(label,[]);groups.get(label).push(Number(r.dice));}if(!groups.size)chart.appendChild(cell('p','No valid Dice observations in this selection.'));for(const [label,values] of groups){const value=values.reduce((a,b)=>a+b,0)/values.length;const row=cell('div',label+' (n='+values.length+'): ');row.className='row';const bar=document.createElement('span');bar.className='bar';bar.style.width=(Math.max(0,Math.min(1,value))*250)+'px';row.appendChild(bar);row.appendChild(cell('span',' '+value.toFixed(4)));chart.appendChild(row);}}
document.getElementById('reset').onclick=()=>{keys.forEach(k=>document.getElementById(k).value='');render();};
document.getElementById('download').onclick=()=>{const escape=x=>{let s=String(x??'');if(/^[=+@\\-\\t\\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};const csv='\\ufeff'+[fields,...current.map(r=>fields.map(f=>r[f]))].map(row=>row.map(escape).join(',')).join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='filtered_results.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.getElementById('metadata').textContent=JSON.stringify(payload.metadata,null,2);render();
</script></main></html>'''
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(template.replace("__DATA__", encoded))
    return str(output)
