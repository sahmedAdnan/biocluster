"""
Self-contained interactive species-review report.

Produces ONE portable review.html that an ecologist can open in any browser
with zero setup, fully offline (GBIF reference panels need internet because
their URLs are kept as links by design). It is the single review report for
the annotation pipeline (stats header, confidence tiers, 4 uncertainty
signals, reference exemplars, runner-up) with the full correction
flexibility:

  - confirm a whole cluster (AI top-1 or runner-up or any other species)
  - relabel individual images, or only some images in a cluster
  - move an image to the correct species
  - rescue an image out of noise into a species
  - mark an image or a cluster as noise or junk
  - assign a species not in the candidate list (free text)
  - undo / redo, autosave to the browser, save/load a session file

On Finalize the browser downloads ONE decisions file:
    file_path,final_label,status
That single file is the entire source of truth. apply_decisions() turns it
into a clean imagefolder dataset through the fixed export_sorted_species().

Reference images come from the pipeline's existing reference_map, keyed by
species name (GBIF run -> GBIF photo URLs kept as links; local/csv run ->
dataset images embedded as base64). 3 references for the AI top-1 species
and 3 for the runner-up.
"""
import base64
import io
import json
import os

import pandas as pd
from PIL import Image

TIER_PRIORITY = {"human_required": 0, "review": 1, "auto_accept": 2}
TIER_LABEL = {"human_required": "Human Required",
              "review": "Review", "auto_accept": "Auto-accepted"}
THUMB_BOX = 250
THUMB_Q = 80
# References are embedded at a higher resolution than cluster thumbnails so
# the ecologist can click to zoom and actually compare morphology. There are
# far fewer unique reference images (a few per species, cached), so the file
# size impact is modest. Lower REF_BOX if the file gets too large.
REF_BOX = 420
REF_Q = 82


def _clean(x):
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def _f(x, d=0.0):
    try:
        v = float(x)
        return d if v != v else v
    except (TypeError, ValueError):
        return d


def _is_url(s):
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


def _data_uri(path, _cache, box=THUMB_BOX, q=THUMB_Q):
    """Embed a local image as a base64 data URI. Cache key includes the box
    so cluster thumbs (small) and reference images (large) never collide."""
    key = (path, box)
    if key in _cache:
        return _cache[key]
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((box, box))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        uri = "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()).decode("ascii")
    except Exception:
        uri = ""
    _cache[key] = uri
    return uri


def _ref_srcs(species, reference_map, _cache, n=3):
    """Return up to n reference images for a species name, as {src, stage}.

    Accepts either a bare path or URL, or a {src, stage} entry as the GBIF
    lookup returns. URLs are passed through as links; local paths are
    embedded. stage is "" for a dataset image, where the stage is whatever
    the collection holds and is not ours to assert."""
    refs = reference_map.get(species, []) if species else []
    out = []
    for r in refs[:n]:
        if isinstance(r, dict):
            src, stage = r.get("src", ""), r.get("stage", "")
        else:
            src, stage = r, ""
        if not src:
            continue
        if not _is_url(src):
            src = _data_uri(src, _cache, REF_BOX, REF_Q)
            if not src:
                continue
        out.append({"src": src, "stage": stage})
    return out


def _majority(vals):
    vals = [_clean(v) for v in vals if _clean(v)]
    if not vals:
        return ""
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def displayed_species(annotations_df) -> list:
    """
    The species names that will get a reference panel: each cluster's top
    choice and its runner-up, which is what build_payload looks up in
    reference_map. Nothing else in the candidate list is ever displayed, so a
    caller assembling reference images only needs these.
    """
    names = []
    for col in ("ai_annotation", "ai_second_choice"):
        if col in annotations_df.columns:
            names += [str(v) for v in annotations_df[col].dropna().unique()]
    skip = {"", "nan", "noise / unknown", "noise", "unknown"}
    return [n for n in dict.fromkeys(names) if n.strip().lower() not in skip]


def build_payload(annotations_df, reference_map, candidate_mode):
    df = annotations_df.copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    cache = {}

    clusters = {}
    for _, r in df.iterrows():
        clusters.setdefault(int(r["cluster_id"]), []).append(r)

    def sort_key(cid):
        if cid == -1:
            return (-1, 0.0)
        rows = clusters[cid]
        tier = _clean(rows[0].get("confidence_tier", "")) or "auto_accept"
        ms = [_f(x.get("margin"), 1.0) for x in rows]
        return (TIER_PRIORITY.get(tier, 3), sum(ms) / len(ms) if ms else 1.0)

    out_clusters = []
    for cid in sorted(clusters, key=sort_key):
        rows = clusters[cid]
        top1 = _majority([x["ai_annotation"] for x in rows])
        top2 = _majority([x.get("ai_second_choice", "") for x in rows])
        ms = [_f(x.get("margin"), None) for x in rows
              if _f(x.get("margin"), None) is not None]
        margin = round(sum(ms) / len(ms), 3) if ms else None

        # Cluster-level signal aggregates: mean of each per-image signal
        # across the cluster's images. Faithful to Methodology Section 11.1,
        # which defines the two confidence meters as cluster-level. The
        # earlier implementation took only the first row, which is one
        # arbitrary image whose values can be unrepresentative (e.g. when
        # the first row of a cluster is an outlier with much higher
        # uncertainty than the other members).
        def _agg(field):
            vs = [_f(x.get(field)) for x in rows]
            return sum(vs) / len(vs) if vs else 0.0
        sem = _agg("semantic_uncertainty")
        spa = _agg("spatial_uncertainty")
        agr = _agg("cluster_agreement")
        tax = _agg("taxonomic_consistency")

        # Cluster tier badge = the worst per-image tier in the cluster, so a
        # cluster shows "human_required" if any image inside it is
        # human_required, "review" if any image is review (and none
        # human_required), otherwise "auto_accept". Conservative; never
        # under-flags. Per-image tier badges still appear on each image
        # card so the expert can see which specific images carry the tier.
        tier_seen = [_clean(x.get("confidence_tier", "")) or "auto_accept"
                     for x in rows]
        tier = min(tier_seen, key=lambda t: TIER_PRIORITY.get(t, 3))

        # A taxonomic_consistency of 0.0 means either "one family, genuinely
        # pure" or "no family resolvable". Only the first is evidence. Where no
        # family was resolved the meter is built from the semantic signal alone,
        # otherwise 40% of the bar comes from a signal that is identically zero
        # and the meter can never fall below 0.4 however bad the name is.
        tax_known = any(str(x.get("taxonomy_informative", "")).strip().lower()
                        in ("true", "1", "1.0") for x in rows)

        cl_conf = round(max(0.0, min(1.0, 1.0 - (spa * 0.5 + agr * 0.5))), 3)
        if tax_known:
            an_conf = round(max(0.0, min(1.0, 1.0 - (sem * 0.6 + tax * 0.4))), 3)
        else:
            an_conf = round(max(0.0, min(1.0, 1.0 - sem)), 3)

        # Per-image annotation confidence = 1 - combined_uncertainty (the same
        # per-image score that sets each image's tier). The cluster average of
        # these confidences is stored so each card can be compared against it,
        # revealing which specific images drag the cluster's confidence down.
        has_cu = "combined_uncertainty" in df.columns
        cus = [_f(x.get("combined_uncertainty")) for x in rows] if has_cu else []
        avg_conf = (round(max(0.0, min(1.0, 1.0 - sum(cus) / len(cus))), 3)
                    if cus else None)

        _SIGKEYS = {"S": "semantic_uncertainty", "P": "spatial_uncertainty",
                    "A": "cluster_agreement", "T": "taxonomic_consistency"}
        images = []
        for x in rows:
            p = x["file_path"]
            cu_i = _f(x.get("combined_uncertainty")) if has_cu else None
            conf_i = (round(max(0.0, min(1.0, 1.0 - cu_i)), 3)
                      if cu_i is not None else None)
            # Dominant uncertainty signal for this image (largest of the 4),
            # so a low-confidence card can say *why* it is hurting the cluster.
            sig_i = {k: _f(x.get(col)) for k, col in _SIGKEYS.items()}
            dom = max(sig_i, key=sig_i.get) if any(sig_i.values()) else ""
            images.append({
                "id": p,
                "name": os.path.basename(p),
                "ai": str(x["ai_annotation"]),
                "ai2": _clean(x.get("ai_second_choice", "")),
                "src": _data_uri(p, cache),
                # Per-image tier (read straight from annotations.csv) so the
                # image card can show its own tier badge. Independent of the
                # cluster's aggregated badge above.
                "tier": (_clean(x.get("confidence_tier", ""))
                         or "auto_accept"),
                # Per-image annotation confidence + dominant hurting signal +
                # the raw per-image S/P/A/T values so the card can show *why*.
                "conf": conf_i,
                "dom": dom,
                "sig": {k: round(v, 2) for k, v in sig_i.items()},
            })

        out_clusters.append({
            "cid": cid, "noise": cid == -1, "n": len(images),
            "tier": tier, "tier_label": TIER_LABEL.get(tier, "Auto-accepted"),
            "margin": margin,
            "sig": {"S": round(sem, 2), "P": round(spa, 2),
                    "A": round(agr, 2), "T": round(tax, 2)},
            "cl_conf": cl_conf, "an_conf": an_conf, "avg_conf": avg_conf,
            "top1": top1, "top2": top2,
            "ref1": _ref_srcs(top1, reference_map, cache),
            "ref2": _ref_srcs(top2, reference_map, cache),
            "images": images,
        })

    tiers = (df["confidence_tier"].value_counts().to_dict()
             if "confidence_tier" in df.columns else {})
    candidates = sorted(
        ({_clean(x) for x in df["ai_annotation"]} |
         {_clean(k) for k in reference_map.keys()}) - {""})
    meta = {
        "total": int(len(df)),
        "n_clusters": int(df.loc[df["cluster_id"] != -1, "cluster_id"].nunique()),
        "auto_accept": int(tiers.get("auto_accept", 0)),
        "review": int(tiers.get("review", 0)),
        "human_required": int(tiers.get("human_required", 0)),
        "mode": candidate_mode,
    }
    return {"clusters": out_clusters, "candidates": candidates, "meta": meta}


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Species Review</title><style>
*{box-sizing:border-box}body{font-family:system-ui,Arial,sans-serif;margin:0;
background:#eef0f3;color:#111}
header{position:sticky;top:0;background:#1f2937;color:#fff;padding:10px 16px;
display:flex;gap:12px;align-items:center;z-index:30;flex-wrap:wrap}
header b{font-size:15px}.muted{font-size:12px;opacity:.85}
button{cursor:pointer;border:1px solid #cbd5e1;background:#fff;border-radius:6px;
padding:5px 10px;font-size:13px}
button.primary{background:#2563eb;color:#fff;border:0}
button.ok{background:#16a34a;color:#fff;border:0}
button.alt{background:#7c3aed;color:#fff;border:0}
button.warn{background:#b91c1c;color:#fff;border:0}
.stats{display:flex;gap:8px;flex-wrap:wrap;padding:10px 14px;background:#fff;
margin:10px 12px 0;border-radius:10px}
.sb{background:#f3f4f6;border-radius:8px;padding:8px 14px;text-align:center;min-width:92px}
.sb .n{font-size:20px;font-weight:700}.sb .l{font-size:11px;color:#555}
.sb.a .n{color:#16a34a}.sb.r .n{color:#d97706}.sb.h .n{color:#dc2626}
.filters{display:flex;gap:8px;padding:10px 14px;flex-wrap:wrap}
.filters button.on{background:#1f2937;color:#fff;border:0}
.legend{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 12px;
padding:8px 14px;background:#fff;border-radius:8px;font-size:11px;color:#374151}
.legend .k{background:#f1f5f9;border-radius:4px;padding:2px 7px}
.legend .k b{color:#991b1b}.legend .note{color:#6b7280}
.legend .drag{color:#dc2626;font-weight:700}
.cluster{background:#fff;margin:12px;border-radius:10px;box-shadow:0 1px 3px #0001;
border-left:6px solid #ccc}
.cluster.t-human_required{border-left-color:#dc2626}
.cluster.t-review{border-left-color:#d97706}
.cluster.t-auto_accept{border-left-color:#16a34a}
.cluster.t-noise{border-left-color:#9a3412;background:#fff7ed}
.chead{display:flex;gap:10px;align-items:center;padding:10px 14px;
border-bottom:1px solid #eee;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.tag.human_required{background:#fee2e2;color:#991b1b}
.tag.review{background:#fef3c7;color:#92400e}
.tag.auto_accept{background:#dcfce7;color:#166534}
.tag.noise{background:#ffedd5;color:#9a3412}
.sig{display:flex;gap:14px;align-items:center;padding:8px 14px;background:#fafafa;
border-bottom:1px solid #f0f0f0;font-size:12px;flex-wrap:wrap}
.chip{background:#eef2ff;color:#3730a3;border-radius:6px;padding:2px 7px;font-weight:600}
.bar{width:110px;height:8px;background:#e5e7eb;border-radius:5px;overflow:hidden;
display:inline-block;vertical-align:middle}.bar i{display:block;height:100%}
.bar i.good{background:#16a34a}.bar i.medium{background:#d97706}.bar i.bad{background:#dc2626}
.refs{display:flex;gap:20px;padding:10px 14px;border-bottom:1px solid #f0f0f0;
flex-wrap:wrap;background:#fcfcfc}
.refbox .t{font-weight:600;margin-bottom:4px;font-size:12px}
.refbox.t1 .t{color:#1d4ed8}.refbox.t2 .t{color:#7c3aed}
.refrow{display:flex;gap:6px;flex-wrap:wrap}
.refrow img{width:155px;height:155px;object-fit:contain;border-radius:4px;
border:1px solid #ddd;background:#fff;cursor:zoom-in}
figure.ref{margin:0;display:flex;flex-direction:column;align-items:center}
figure.ref figcaption{font-size:10px;color:#555;margin-top:2px;max-width:155px;
text-align:center}
figure.ref figcaption.unk{color:#b45309;font-style:italic}
.noref{font-size:11px;color:#999;padding:6px}
.cbody{display:flex;flex-wrap:wrap;gap:10px;padding:12px}
.card{width:180px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}
.card img{width:180px;height:165px;object-fit:cover;display:block;background:#eee;
cursor:zoom-in}
.card .meta{padding:6px;font-size:11px}
.card input{width:100%;font-size:12px;padding:3px;border:1px solid #cbd5e1;
border-radius:4px;margin-top:3px}
.badge{font-size:10px;padding:2px 6px;border-radius:10px}
.b-ai{background:#e0e7ff;color:#3730a3}.b-human{background:#dcfce7;color:#166534}
.b-noise{background:#ffedd5;color:#9a3412}.b-junk{background:#fee2e2;color:#991b1b}
/* Per-image tier badges (traffic-light palette matching Figure 5.9). */
.b-tier-r{background:#fef3c7;color:#854d0e}
.b-tier-h{background:#fee2e2;color:#991b1b}
/* Per-image annotation confidence bar on each card. */
.iconf{display:flex;align-items:center;gap:4px;margin-top:4px;font-size:10px;color:#374151}
.iconf .bar{width:66px;height:6px}
.iconf .drag{color:#dc2626;font-weight:700;cursor:help}
.iconf .dom{color:#9ca3af}
/* Per-image S/P/A/T signal chips; the dominant (hurting) one is highlighted. */
.isig{display:flex;flex-wrap:wrap;gap:3px;margin-top:3px;font-size:9px;color:#6b7280}
.isig span{background:#f1f5f9;border-radius:4px;padding:1px 4px;cursor:help}
.isig span.hot{background:#fee2e2;color:#991b1b;font-weight:700}
.qrow{display:flex;gap:3px;margin-top:4px}.qrow button{flex:1;padding:3px;font-size:10px}
.collapsed .sig,.collapsed .refs,.collapsed .cbody{display:none}
#zoom{position:fixed;inset:0;background:#000c;display:none;align-items:center;
justify-content:center;z-index:50}#zoom img{max-width:92vw;max-height:92vh;border-radius:6px}
</style></head><body>
<header><b>Species Review</b><span class="muted" id="stat"></span>
<span style="flex:1"></span>
<button data-act="undo" id="ub">Undo</button>
<button data-act="redo" id="rb">Redo</button>
<button data-act="savesession">Save session</button>
<button data-act="loadsession">Load session</button>
<input type="file" id="lf" style="display:none" accept=".json">
<button class="primary" data-act="finalize">Finalize &amp; Download</button>
</header>
<div class="stats" id="stats"></div>
<div class="filters" id="filters">
<button class="on" data-act="filter" data-f="all">All</button>
<button data-act="filter" data-f="human_required">Human Required</button>
<button data-act="filter" data-f="review">Review</button>
<button data-act="filter" data-f="auto_accept">Auto-accepted</button>
<button data-act="filter" data-f="noise">Noise</button>
</div>
<div class="legend">
<b>Per-image signals</b> <span class="note">(0-1, higher = more uncertain):</span>
<span class="k"><b>S</b> semantic (text match)</span>
<span class="k"><b>P</b> spatial (cluster boundary)</span>
<span class="k"><b>A</b> agreement (vote vs cluster)</span>
<span class="k"><b>T</b> taxonomic (family mix)</span>
<span class="note">conf = 1 &minus; combined uncertainty &nbsp;|&nbsp;
 <span class="drag">&#9660; drags</span> = below cluster average</span>
</div>
<div id="app"></div><datalist id="cands"></datalist>
<div id="zoom"><img id="zimg"></div>
<script>
const DATA=__PAYLOAD__;
// Discriminate the localStorage key by a stable hash of the dataset (first +
// last image id + total). Two pipelines that produce the same image_count +
// cluster_count for the same dataset can still differ on image-id strings
// (e.g. path-separator quirks), and the old "<total>_<n_clusters>" key
// collided across them, replacing ST with incompatible image-ids.
function _hash(s){let h=0;for(let i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return Math.abs(h).toString(36);}
const _allClusters=DATA.clusters||[];
const _firstC=_allClusters[0];
const _lastC=_allClusters[_allClusters.length-1];
const _firstId=(_firstC&&_firstC.images&&_firstC.images[0])?_firstC.images[0].id:'';
const _lastImgs=(_lastC&&_lastC.images)||[];
const _lastId=_lastImgs.length?_lastImgs[_lastImgs.length-1].id:'';
const KEY='speciesreview_'+DATA.meta.total+'_'+DATA.meta.n_clusters+'_'+_hash(_firstId+'|'+_lastId);
let ST={},FILTER='all',hist=[],fut=[];
// EXP tracks which cluster ids the user has expanded so that render() (which
// is called after every state change) preserves the open/closed state across
// re-renders. FIRSTRENDER is true only for the very first render() call, on
// which we auto-expand the first visible cluster as a visual cue.
let EXP=new Set(),FIRSTRENDER=true;
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;',
'>':'&gt;','"':'&quot;'}[c]))}
function cls(v){return v>=0.7?'good':v>=0.4?'medium':'bad'}
// Idempotent: only initialise images that are missing from ST. Safe to call
// after restore() to fill in any image ids that the saved session does not
// cover (e.g. when localStorage was written by an earlier run that used a
// different image-id format).
function initST(){DATA.clusters.forEach(c=>c.images.forEach(im=>{
 if(!ST[im.id]) ST[im.id]={final:im.ai,status:c.noise?'noise':'species'};
}));}
function snapshot(){hist.push(JSON.stringify(ST));if(hist.length>200)hist.shift();
 fut=[];persist();}
function undo(){if(!hist.length)return;fut.push(JSON.stringify(ST));
 ST=JSON.parse(hist.pop());persist();render();}
function redo(){if(!fut.length)return;hist.push(JSON.stringify(ST));
 ST=JSON.parse(fut.pop());persist();render();}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ST));}catch(e){}}
function restore(){try{const s=localStorage.getItem(KEY);
 if(s){ST=JSON.parse(s);return true;}}catch(e){}return false;}
function setFilter(f,b){FILTER=f;
 document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');render();}
function vis(c){if(FILTER==='all')return true;
 if(FILTER==='noise')return c.noise;return !c.noise&&c.tier===FILTER;}
function refbox(k,lab,name,srcs){if(!name)return '';
 // loading="lazy" intentionally OFF: it's a no-op for data URIs (no network
 // fetch to defer) and some Chromium versions silently skip the base64
 // decode when lazy is set on a data URI, leaving blank image boxes.
 // The stage caption matters: GBIF media for an adult species is often a
 // larva, so a reference the reviewer cannot stage is not a reference.
 const body=srcs.length?srcs.map(s=>`<figure class="ref"><img
   src="${esc(s.src)}" data-act="zoom">${s.stage?
   `<figcaption class="${s.stage==='stage not recorded'?'unk':''}"
   >${esc(s.stage)}</figcaption>`:''}</figure>`).join('')
  :'<div class="noref">no reference for this species</div>';
 return `<div class="refbox ${k}"><div class="t">${lab}: ${esc(name)}</div>
  <div class="refrow">${body}</div></div>`;}
function render(){
 const a=document.getElementById('app');a.innerHTML='';
 let _firstVisible=true;
 DATA.clusters.forEach(c=>{if(!vis(c))return;
  const d=document.createElement('div');
  // Start every cluster collapsed except (a) the first visible cluster on the
  // very first render() call, as a visual cue, or (b) any cluster the user has
  // explicitly expanded since then (tracked in EXP). Image cards are only
  // built on first expand; without this a dataset like CUB-200 builds 11,788
  // base64 <img> nodes (~108 MB of DOM strings) at page load and the browser
  // stalls before rendering.
  let _isExpanded;
  if(FIRSTRENDER){
   _isExpanded=_firstVisible;
   if(_isExpanded)EXP.add(c.cid);
  }else{
   _isExpanded=EXP.has(c.cid);
  }
  _firstVisible=false;
  const _initiallyCollapsed=!_isExpanded;
  d.className='cluster t-'+(c.noise?'noise':c.tier)+(_initiallyCollapsed?' collapsed':'');
  const title=c.noise?'NOISE (unassigned)':'Cluster '+c.cid;
  const dm=c.margin!=null?' &Delta;'+c.margin:'';
  const tg=c.noise?'noise':c.tier,tl=c.noise?'Noise':c.tier_label;
  let h=`<div class="chead">
   <button data-act="toggle">&#9662;</button>
   <b>${title}</b><span class="tag ${tg}">${tl}</span>
   <span class="muted">${c.n} images</span>`;
  if(!c.noise){
   h+=`<button class="ok" data-act="applyC" data-cid="${c.cid}"
     data-val="${esc(c.top1)}">Confirm: ${esc(c.top1)}</button>`;
   if(c.top2)h+=`<button class="alt" data-act="applyC" data-cid="${c.cid}"
     data-val="${esc(c.top2)}">Runner-up: ${esc(c.top2)}${dm}</button>`;
   h+=`<input list="cands" placeholder="other species..." style="width:180px"
     id="ci${c.cid}"><button data-act="applyTyped" data-cid="${c.cid}">Apply</button>`;
  }
  h+=`<button class="warn" data-act="markC" data-cid="${c.cid}" data-val="noise">
     All noise</button>
   <button class="warn" data-act="markC" data-cid="${c.cid}" data-val="junk">
     All junk</button></div>`;
  let s='';
  if(!c.noise)s=`<div class="sig"><span>Signals (cluster mean):</span>
   <span class="chip" title="Semantic uncertainty">S ${c.sig.S}</span>
   <span class="chip" title="Spatial uncertainty">P ${c.sig.P}</span>
   <span class="chip" title="Cluster agreement">A ${c.sig.A}</span>
   <span class="chip" title="Taxonomic consistency">T ${c.sig.T}</span>
   <span style="margin-left:8px" title="Cluster cohesion (spatial + agreement).
Per-image annotation confidence is shown on each image card below.">Cluster conf
    <span class="bar">
    <i class="${cls(c.cl_conf)}" style="width:${Math.round(c.cl_conf*100)}%"></i>
    </span> ${Math.round(c.cl_conf*100)}%</span></div>`;
  let rf='';
  if(!c.noise)rf=`<div class="refs">
   ${refbox('t1','Reference (AI top-1)',c.top1,c.ref1)}
   ${refbox('t2','Reference (runner-up)',c.top2,c.ref2)}</div>`;
  d.innerHTML=h+s+rf+'<div class="cbody"></div>';
  const b=d.querySelector('.cbody');
  // Stash the cluster's images on the DOM node so hydrate() can build the
  // image cards on demand the first time the cluster is expanded.
  d._cluster=c;
  d._hydrated=false;
  if(!_initiallyCollapsed){
   c.images.forEach(im=>b.appendChild(card(im,c)));
   d._hydrated=true;
  }
  a.appendChild(d);
 });
 FIRSTRENDER=false;
 updateStat();
}
function card(im,c){
 const st=ST[im.id];const el=document.createElement('div');el.className='card';
 const bd=st.status==='noise'?'<span class="badge b-noise">noise</span>':
  st.status==='junk'?'<span class="badge b-junk">junk</span>':
  (st.final!==im.ai?'<span class="badge b-human">human</span>':
   '<span class="badge b-ai">ai</span>');
 // Per-image tier badge so the expert can see, inside a cluster, exactly
 // which images carry the human_required / review tier. auto_accept is
 // implicit (no badge) to keep the card uncluttered.
 const tb=im.tier==='human_required'?'<span class="badge b-tier-h" title="This image’s confidence tier is human_required">need human</span>':
  im.tier==='review'?'<span class="badge b-tier-r" title="This image’s confidence tier is review">review</span>':'';
 let q=`<button data-act="setI" data-id="${esc(im.id)}" data-val="${esc(im.ai)}"
   title="${esc(im.ai)}">&#10003; top-1</button>`;
 if(im.ai2)q+=`<button data-act="setI" data-id="${esc(im.id)}"
   data-val="${esc(im.ai2)}" title="${esc(im.ai2)}">&#8594; runner-up</button>`;
 // Per-image annotation confidence (1 - combined uncertainty). A red "drags"
 // marker appears when this image is below its cluster's average confidence,
 // i.e. it is one of the images pulling the annotation confidence down, with
 // the dominant hurting signal (S/P/A/T) named in the tooltip.
 const DOMNAME={S:'semantic (text match)',P:'spatial (cluster boundary)',
  A:'cluster agreement',T:'taxonomic consistency'};
 let cf='';
 if(im.conf!=null){
  const pct=Math.round(im.conf*100);
  const below=c.avg_conf!=null&&im.conf<c.avg_conf;
  const why=DOMNAME[im.dom]||'';
  cf=`<div class="iconf" title="Annotation confidence for this image = 1 - combined uncertainty">
   conf <span class="bar"><i class="${cls(im.conf)}" style="width:${pct}%"></i></span> ${pct}%`
   +(below?` <span class="drag" title="Below the cluster average of ${Math.round(c.avg_conf*100)}%. This image lowers the cluster's annotation confidence${why?'; main issue: '+why:''}.">&#9660; drags</span>`:'')
   +`</div>`;
 }
 // Per-image S/P/A/T breakdown (each in [0,1]; higher = more uncertain). The
 // dominant signal is highlighted so you can read at a glance why this image's
 // confidence is what it is.
 let sg='';
 if(im.sig){
  sg='<div class="isig">'+['S','P','A','T'].map(k=>{
    const v=im.sig[k];const hot=(k===im.dom)?' hot':'';
    return `<span class="${hot}" title="${DOMNAME[k]} uncertainty">${k} ${v.toFixed(2)}</span>`;
  }).join('')+'</div>';
 }
 el.innerHTML=`<img src="${esc(im.src)}" data-act="zoom">

  <div class="meta"><div title="${esc(im.name)}">${esc(im.name).slice(0,28)}</div>
  <div>${bd}${tb?' '+tb:''} <b>${esc(st.final)}</b></div>
  ${cf}${sg}
  <div class="qrow">${q}</div>
  <input class="sp-input" list="cands" value="${esc(st.final)}"
   data-id="${esc(im.id)}">
  <div class="qrow">
   <button data-act="setS" data-id="${esc(im.id)}" data-val="noise">noise</button>
   <button data-act="setS" data-id="${esc(im.id)}" data-val="junk">junk</button>
   <button data-act="setI" data-id="${esc(im.id)}" data-val="${esc(im.ai)}">reset</button>
  </div></div>`;
 return el;
}
function zoom(s){document.getElementById('zimg').src=s;
 document.getElementById('zoom').style.display='flex';}
function setI(id,v){if(!v)return;snapshot();ST[id]={final:v,status:'species'};render();}
function setS(id,s){snapshot();ST[id].status=s;render();}
function applyC(cid,v){snapshot();
 DATA.clusters.find(c=>c.cid===cid).images.forEach(im=>{
  ST[im.id]={final:v,status:'species'};});render();}
function applyTyped(cid){const el=document.getElementById('ci'+cid);
 const v=el?el.value.trim():'';if(v)applyC(cid,v);}
function markC(cid,s){snapshot();
 DATA.clusters.find(c=>c.cid===cid).images.forEach(im=>{
  ST[im.id].status=s;});render();}
function updateStat(){let h=0,n=0,j=0;
 for(const k in ST){const v=ST[k];const im=IMG[k];
  if(v.status==='noise')n++;else if(v.status==='junk')j++;
  else if(im&&v.final!==im.ai)h++;}
 document.getElementById('stat').textContent=
  `${h} human-corrected | ${n} noise | ${j} junk`;
 document.getElementById('ub').disabled=!hist.length;
 document.getElementById('rb').disabled=!fut.length;}
const IMG={};DATA.clusters.forEach(c=>c.images.forEach(im=>IMG[im.id]=im));
function csvCell(s){s=(s==null?'':String(s));
 return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function finalize(){
 const lines=['file_path,final_label,status'];
 for(const k in ST){const v=ST[k];
  lines.push([csvCell(k),csvCell(v.final),csvCell(v.status)].join(','));}
 const blob=new Blob([lines.join('\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='decisions.csv';a.click();
 alert('decisions.csv downloaded. Send this one file back.');
}
function saveSession(){const blob=new Blob([JSON.stringify({KEY:KEY,ST:ST})],
 {type:'application/json'});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='review_session.json';a.click();}
function loadSession(ev){const f=ev.target.files[0];if(!f)return;
 const r=new FileReader();r.onload=()=>{try{const o=JSON.parse(r.result);
  if(o.ST){snapshot();ST=o.ST;render();alert('Session loaded.');}}
  catch(e){alert('Invalid session file.');}};r.readAsText(f);}
document.getElementById('cands').innerHTML=
 DATA.candidates.map(c=>`<option value="${esc(c)}">`).join('');
const m=DATA.meta;
document.getElementById('stats').innerHTML=
 `<div class="sb"><div class="n">${m.total}</div><div class="l">Images</div></div>
  <div class="sb"><div class="n">${m.n_clusters}</div><div class="l">Clusters</div></div>
  <div class="sb a"><div class="n">${m.auto_accept}</div><div class="l">Auto-accepted</div></div>
  <div class="sb r"><div class="n">${m.review}</div><div class="l">Review</div></div>
  <div class="sb h"><div class="n">${m.human_required}</div><div class="l">Human Required</div></div>
  <div class="sb"><div class="n" style="font-size:14px">${esc(m.mode)}</div>
   <div class="l">Mode</div></div>`;
// ---- Event delegation: one click + one change listener, no inline handlers ----
document.addEventListener('click',e=>{
 if(e.target.id==='zoom'||e.target.id==='zimg'){
  document.getElementById('zoom').style.display='none';return;}
 const t=e.target.closest('[data-act]');if(!t)return;
 const a=t.dataset.act;
 if(a==='toggle'){
  const cdiv=t.closest('.cluster');
  const cid=cdiv._cluster?cdiv._cluster.cid:null;
  // Hydrate image cards on first expand (lazy DOM build for large datasets).
  if(cdiv.classList.contains('collapsed')&&!cdiv._hydrated){
   const b=cdiv.querySelector('.cbody');
   cdiv._cluster.images.forEach(im=>b.appendChild(card(im,cdiv._cluster)));
   cdiv._hydrated=true;
  }
  cdiv.classList.toggle('collapsed');
  // Keep EXP in sync so the next render() preserves the user's expand state.
  if(cid!==null){
   if(cdiv.classList.contains('collapsed'))EXP.delete(cid);else EXP.add(cid);
  }
  return;
 }
 if(a==='zoom'){zoom(t.getAttribute('src'));return;}
 if(a==='filter'){setFilter(t.dataset.f,t);return;}
 if(a==='undo'){undo();return;}
 if(a==='redo'){redo();return;}
 if(a==='finalize'){finalize();return;}
 if(a==='savesession'){saveSession();return;}
 if(a==='loadsession'){document.getElementById('lf').click();return;}
 if(a==='applyC'){applyC(parseInt(t.dataset.cid,10),t.dataset.val);return;}
 if(a==='applyTyped'){applyTyped(parseInt(t.dataset.cid,10));return;}
 if(a==='markC'){markC(parseInt(t.dataset.cid,10),t.dataset.val);return;}
 if(a==='setI'){setI(t.dataset.id,t.dataset.val);return;}
 if(a==='setS'){setS(t.dataset.id,t.dataset.val);return;}
});
document.addEventListener('change',e=>{
 if(e.target.classList&&e.target.classList.contains('sp-input')){
  setI(e.target.dataset.id,e.target.value);return;}
 if(e.target.id==='lf'){loadSession(e);return;}
});
initST();restore();initST();
document.addEventListener('keydown',e=>{
 if(e.ctrlKey&&e.key==='z'){e.preventDefault();undo();}
 if(e.ctrlKey&&(e.key==='y'||(e.shiftKey&&e.key==='Z'))){e.preventDefault();redo();}
 if(e.key==='Escape')document.getElementById('zoom').style.display='none';});
render();
</script></body></html>"""


def generate_interactive_report(annotations_df, reference_map, out_path,
                                config=None, candidate_mode="gbif",
                                **_ignored):
    """Write a single self-contained interactive review.html."""
    payload = build_payload(annotations_df, reference_map, candidate_mode)
    # json.dumps defaults to ensure_ascii=True, so all non-ASCII
    # (including U+2028/U+2029) is already escaped. The only residual
    # breakout risk is a literal close-script tag inside a string.
    payload_json = json.dumps(payload).replace("</", r"<\/")
    html = _HTML.replace("__PAYLOAD__", payload_json)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[interactive] {out_path}  ({size_mb:.1f} MB, "
          f"{len(payload['clusters'])} clusters, {payload['meta']['total']} images)")
    return out_path


def apply_decisions(decisions_csv, annotations_df):
    """Apply a single decisions.csv (file_path,final_label,status) onto the
    master annotations, mapping status to the fields export_sorted_species
    routes on. Returns the final DataFrame ready for export."""
    dec = pd.read_csv(decisions_csv)
    need = {"file_path", "final_label", "status"}
    if not need.issubset(dec.columns):
        raise ValueError(
            f"decisions CSV must have columns {sorted(need)}; "
            f"got {list(dec.columns)}")
    dmap = {r["file_path"]: (str(r["final_label"]), str(r["status"]).strip())
            for _, r in dec.iterrows()}

    df = annotations_df.copy()
    src_col = "annotation_source"
    if src_col not in df.columns:
        df[src_col] = "AI_ZeroShot"

    for i, row in df.iterrows():
        p = row["file_path"]
        if p not in dmap:
            continue
        final, status = dmap[p]
        orig = str(row["ai_annotation"])
        if status == "junk":
            df.at[i, src_col] = "Flagged_for_Deletion"
        elif status == "noise":
            df.at[i, src_col] = "Human_Noise"
        else:  # species
            df.at[i, "ai_annotation"] = final
            df.at[i, src_col] = ("Human_Expert" if final != orig
                                 else "Human_Confirmed")
    return df
