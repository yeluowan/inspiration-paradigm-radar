"""内容范式雷达 — Cowork Guard 子应用.

前置口径 → 常青词分层 → 三路特征聚类 → 双通道成簇 → 语义复核 → 四区呈现。
持久化：PostgreSQL via db.properties（数据集 + 每日快照，用于范式演进追踪）。
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

import engine
import nebula

BASE = Path(__file__).resolve().parent


# ───────────────────────── SSO ─────────────────────────

def _parse_sso_user(decrypted_userinfo: Optional[str]) -> Optional[dict]:
    if not decrypted_userinfo:
        return None
    try:
        data = json.loads(decrypted_userinfo.encode("latin-1").decode("utf-8"))
    except Exception:
        return None
    return {
        "email": data.get("email") or data.get("workEmail"),
        "name": data.get("name") or data.get("displayName") or data.get("username"),
        "userId": data.get("userId") or data.get("id"),
    }


def _require_user(decrypted_userinfo: Optional[str]) -> dict:
    """Use SSO when deployed internally; allow a local demo user by default.

    Set REQUIRE_SSO=true in production if requests must carry Decrypted-Userinfo.
    """
    user = _parse_sso_user(decrypted_userinfo)
    if user:
        return user
    if os.getenv("REQUIRE_SSO", "false").lower() == "true":
        raise HTTPException(status_code=401, detail="unauthenticated")
    return {"email": "local@example.com", "name": "Local user", "userId": "local"}


# ───────────────────────── DB ─────────────────────────

def _load_db_props(path: str = "db.properties") -> dict[str, str]:
    p = BASE / path
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


_DDL = """
CREATE TABLE IF NOT EXISTS pr_dataset (
  id SERIAL PRIMARY KEY,
  owner_email TEXT NOT NULL,
  name TEXT NOT NULL,
  vertical TEXT,
  period TEXT,
  notes JSONB NOT NULL,
  params JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pr_dataset_owner ON pr_dataset(owner_email);

CREATE TABLE IF NOT EXISTS pr_snapshot (
  id SERIAL PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES pr_dataset(id) ON DELETE CASCADE,
  snap_date DATE NOT NULL,
  overview JSONB NOT NULL,
  clusters JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (dataset_id, snap_date)
);
""" + nebula.DDL

_db_ready = False


def _conn():
    props = _load_db_props()
    if not props.get("db.host"):
        raise HTTPException(503, "数据库未配置（平台未注入 db.properties），保存/快照功能不可用；即席分析仍可正常使用。")
    import psycopg
    return psycopg.connect(
        host=props["db.host"], port=int(props["db.port"]),
        user=props["db.username"], password=props["db.password"],
        dbname=props["db.database"], connect_timeout=8,
    )


def _db():
    global _db_ready
    conn = _conn()
    if not _db_ready:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        _db_ready = True
    return conn


# ───────────────────────── App ─────────────────────────

app = FastAPI(title="内容范式雷达")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def index() -> RedirectResponse:
    """单一入口：统一进星图，顶栏切换范式雷达。"""
    return RedirectResponse(url="nebula", status_code=307)


@app.get("/legacy")
def legacy() -> FileResponse:
    """旧版自带聚类工作台（已被外部范式雷达取代，保留备查）。"""
    return FileResponse(BASE / "static" / "index.html")


@app.get("/nebula")
def nebula_page() -> FileResponse:
    return FileResponse(BASE / "static" / "nebula.html")


@app.get("/api/me")
def me(decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    props = _load_db_props()
    return {"user": u, "db": bool(props.get("db.host"))}


@app.get("/api/config")
def config(decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    _require_user(decrypted_userinfo)
    return {
        "stopwords": engine.DEFAULT_STOPWORDS,
        "protected": engine.DEFAULT_PROTECTED,
        "synonyms": engine.DEFAULT_SYNONYMS,
        "rules": [{"key": k, "label": l, "words": w} for k, l, w in engine.NON_PARADIGM_RULES],
    }


@app.get("/api/demo")
def demo(decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    _require_user(decrypted_userinfo)
    return {"notes": engine.demo_notes()}


@app.post("/api/analyze")
def analyze(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> JSONResponse:
    _require_user(decrypted_userinfo)
    notes = payload.get("notes") or []
    if not notes:
        raise HTTPException(400, "notes 为空")
    p = payload.get("params") or {}
    result = engine.analyze(
        notes,
        min_likes=int(p.get("min_likes", 1000)),
        burst_likes=int(p.get("burst_likes", 10000)),
        sim_threshold=float(p.get("sim_threshold", 0.22)),
        like_rate_gate=float(p.get("like_rate_gate", 0.10)),
        breakout_gate=float(p.get("breakout_gate", 3.0)),
        stopwords=p.get("stopwords") or None,
        protected=p.get("protected") or None,
        synonyms=p.get("synonyms") or None,
    )
    return JSONResponse(result)


@app.post("/api/parse-csv")
async def parse_csv(
    file: UploadFile = File(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    _require_user(decrypted_userinfo)
    raw = await file.read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        raise HTTPException(400, "文件编码无法识别，请存为 UTF-8 CSV")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise HTTPException(400, "CSV 无数据行")
    return {"notes": rows, "count": len(rows), "columns": list(rows[0].keys())}


# ── 数据集 ──

@app.get("/api/datasets")
def list_datasets(decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT d.id, d.name, d.vertical, d.period, jsonb_array_length(d.notes),
                      d.updated_at, (SELECT COUNT(*) FROM pr_snapshot s WHERE s.dataset_id=d.id)
               FROM pr_dataset d WHERE d.owner_email=%s ORDER BY d.updated_at DESC""",
            (u["email"],),
        )
        rows = cur.fetchall()
    return {"datasets": [
        {"id": r[0], "name": r[1], "vertical": r[2], "period": r[3],
         "note_count": r[4], "updated_at": str(r[5]), "snapshots": r[6]} for r in rows
    ]}


@app.post("/api/datasets")
def save_dataset(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    u = _require_user(decrypted_userinfo)
    name = (payload.get("name") or "").strip()
    notes = payload.get("notes") or []
    if not name or not notes:
        raise HTTPException(400, "name / notes 必填")
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO pr_dataset (owner_email, name, vertical, period, notes, params)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
            (u["email"], name, payload.get("vertical", ""), payload.get("period", ""),
             json.dumps(notes, ensure_ascii=False), json.dumps(payload.get("params") or {}, ensure_ascii=False)),
        )
        did = cur.fetchone()[0]
        conn.commit()
    return {"id": did}


@app.get("/api/datasets/{did}")
def get_dataset(did: int, decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, vertical, period, notes, params FROM pr_dataset WHERE id=%s AND owner_email=%s",
                    (did, u["email"]))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "数据集不存在")
    return {"name": row[0], "vertical": row[1], "period": row[2], "notes": row[3], "params": row[4]}


@app.delete("/api/datasets/{did}")
def del_dataset(did: int, decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM pr_dataset WHERE id=%s AND owner_email=%s", (did, u["email"]))
        conn.commit()
    return {"ok": True}


# ── 每日快照 / 演进追踪 ──

@app.post("/api/datasets/{did}/snapshot")
def make_snapshot(did: int, decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT notes, params FROM pr_dataset WHERE id=%s AND owner_email=%s", (did, u["email"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "数据集不存在")
        p = row[1] or {}
        result = engine.analyze(
            row[0],
            min_likes=int(p.get("min_likes", 1000)),
            burst_likes=int(p.get("burst_likes", 10000)),
            sim_threshold=float(p.get("sim_threshold", 0.22)),
        )
        slim = [{"name": c["name"], "note_count": c["note_count"], "author_count": c["author_count"],
                 "likes_median": c["likes_median"], "worth_doing": c["worth_doing"], "bucket": b}
                for b in ("core", "review", "burst_candidate") for c in result[b]]
        cur.execute(
            """INSERT INTO pr_snapshot (dataset_id, snap_date, overview, clusters)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (dataset_id, snap_date) DO UPDATE
                 SET overview=EXCLUDED.overview, clusters=EXCLUDED.clusters, created_at=NOW()""",
            (did, date.today(), json.dumps(result["overview"], ensure_ascii=False),
             json.dumps(slim, ensure_ascii=False)),
        )
        conn.commit()
    return {"ok": True, "snap_date": str(date.today()), "overview": result["overview"]}


@app.get("/api/datasets/{did}/trend")
def trend(did: int, decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    u = _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pr_dataset WHERE id=%s AND owner_email=%s", (did, u["email"]))
        if not cur.fetchone():
            raise HTTPException(404, "数据集不存在")
        cur.execute(
            "SELECT snap_date, overview, clusters FROM pr_snapshot WHERE dataset_id=%s ORDER BY snap_date DESC LIMIT 30",
            (did,),
        )
        rows = cur.fetchall()
    snaps = [{"date": str(r[0]), "overview": r[1], "clusters": r[2]} for r in rows]
    diff: dict[str, Any] = {"new": [], "grown": [], "faded": []}
    if len(snaps) >= 2:
        cur_map = {c["name"]: c for c in snaps[0]["clusters"]}
        prev_map = {c["name"]: c for c in snaps[1]["clusters"]}
        for name, c in cur_map.items():
            if name not in prev_map:
                diff["new"].append(c)
            elif c["note_count"] > prev_map[name]["note_count"]:
                diff["grown"].append({**c, "delta": c["note_count"] - prev_map[name]["note_count"]})
        for name, c in prev_map.items():
            if name not in cur_map:
                diff["faded"].append(c)
    return {"snapshots": snaps, "diff": diff}


# ═════════════ 星图灵感雷达 ═════════════

def _load_ai_props() -> dict[str, str]:
    p = BASE / "ai.properties"
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _star_row(r) -> dict:
    return {
        "id": r[0], "title": r[1], "layer": r[2], "source": r[3], "url": r[4] or "",
        "mechanism": r[5] or "", "topics": r[6] or [], "emotions": r[7] or [],
        "audience": r[8] or "", "note_count": r[9], "author_count": r[10],
        "likes": r[11], "views": r[12], "fans": r[13], "samples": r[14] or [],
        "week": r[15] or "", "report": r[16], "created_by": r[17] or "",
    }


_STAR_COLS = ("star_key,title,layer,source,url,mechanism,topics,emotions,audience,"
              "note_count,author_count,likes,views,fans,samples,week,report,created_by")


@app.get("/api/nebula/graph")
def nebula_graph(
    week: str = "",
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    _require_user(decrypted_userinfo)
    try:
        with _db() as conn, conn.cursor() as cur:
            if week:
                cur.execute(f"SELECT {_STAR_COLS} FROM neb_star WHERE week=%s ORDER BY id", (week,))
            else:
                cur.execute(f"SELECT {_STAR_COLS} FROM neb_star ORDER BY id")
            stars = [_star_row(r) for r in cur.fetchall()]
            cur.execute("SELECT a,b,strength,kinds,reason FROM neb_rel")
            rels = [{"a": r[0], "b": r[1], "strength": r[2], "kinds": r[3] or [], "reason": r[4] or ""}
                    for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT week FROM neb_star WHERE week<>'' ORDER BY week DESC")
            weeks = [r[0] for r in cur.fetchall() if r[0]]
    except HTTPException:
        stars, rels, weeks = [], [], []
    if not stars:
        stars, rels = nebula.demo_graph()
        weeks = sorted({s["week"] for s in stars if s.get("week")}, reverse=True)
        demo = True
    else:
        demo = False
    keys = {s["id"] for s in stars}
    rels = [r for r in rels if r["a"] in keys and r["b"] in keys]
    for s in stars:
        s.update({k: v for k, v in nebula.score_star(s).items() if k in ("score", "like_rate", "breakout")})
    return {"stars": stars, "rels": rels, "weeks": weeks, "demo": demo,
            "ai": bool(_load_ai_props().get("ai.base_url"))}


@app.post("/api/nebula/star")
def nebula_add_star(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    """人工拖入一颗灵感星（灵感 / 未过会选题 / 刷到的好范式）。"""
    u = _require_user(decrypted_userinfo)
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "title 必填")
    key = payload.get("id") or ("i" + str(abs(hash(title + str(u["email"]))))[:10])
    topics = payload.get("topics") or []
    mech = payload.get("mechanism") or payload.get("desc") or ""
    emotions = payload.get("emotions") or nebula.detect_emotion(title + mech)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO neb_star (star_key,title,layer,source,url,mechanism,topics,emotions,
                                     audience,note_count,author_count,likes,views,fans,samples,week,created_by)
               VALUES (%s,%s,'insp',%s,%s,%s,%s,%s,%s,1,1,%s,%s,%s,'[]'::jsonb,%s,%s)
               ON CONFLICT (star_key) DO UPDATE SET title=EXCLUDED.title, mechanism=EXCLUDED.mechanism,
                 topics=EXCLUDED.topics, emotions=EXCLUDED.emotions, url=EXCLUDED.url, updated_at=NOW()""",
            (key, title, payload.get("source", "manual"), payload.get("url", ""), mech,
             json.dumps(topics, ensure_ascii=False), json.dumps(emotions, ensure_ascii=False),
             payload.get("audience", ""), int(payload.get("likes", 0)), int(payload.get("views", 0)),
             int(payload.get("fans", 0)), payload.get("week", ""), u["email"]),
        )
        for rid in (payload.get("relatedIds") or []):
            a, b = sorted([key, rid])
            cur.execute(
                """INSERT INTO neb_rel (a,b,strength,kinds,reason) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (a,b) DO UPDATE SET strength=EXCLUDED.strength, reason=EXCLUDED.reason""",
                (a, b, 0.62, json.dumps(["manual"], ensure_ascii=False), "人工标注的关联"),
            )
        conn.commit()
    return {"ok": True, "id": key}


@app.delete("/api/nebula/star/{key}")
def nebula_del_star(key: str, decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM neb_star WHERE star_key=%s", (key,))
        cur.execute("DELETE FROM neb_rel WHERE a=%s OR b=%s", (key, key))
        conn.commit()
    return {"ok": True}


@app.post("/api/nebula/ingest")
def nebula_ingest(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    """给上游 AI 周更程序的写入口。

    body: {week, stars:[{id,title,layer?,url,mechanism,topics[],emotions[],audience,
                         note_count,author_count,likes,views,fans,samples[]}],
           relations?: [{a,b,strength,kinds[],reason}], autolink?: true}
    """
    u = _require_user(decrypted_userinfo)
    stars = payload.get("stars") or []
    if not stars:
        raise HTTPException(400, "stars 为空")
    week = payload.get("week", "")
    norm = []
    with _db() as conn, conn.cursor() as cur:
        for s in stars:
            key = str(s.get("id") or s.get("note_id") or "").strip()
            title = (s.get("title") or "").strip()
            if not key or not title:
                continue
            mech = s.get("mechanism", "")
            emo = s.get("emotions") or nebula.detect_emotion(title + " " + mech)
            rec = {
                "id": key, "title": title, "layer": s.get("layer") or "motif",
                "mechanism": mech, "topics": s.get("topics") or [], "emotions": emo,
                "audience": s.get("audience", ""),
                "note_count": int(s.get("note_count", 1)), "author_count": int(s.get("author_count", 1)),
                "likes": int(s.get("likes", 0)), "views": int(s.get("views", 0)),
                "fans": int(s.get("fans", 0)), "samples": s.get("samples") or [],
            }
            norm.append(rec)
            cur.execute(
                """INSERT INTO neb_star (star_key,title,layer,source,url,mechanism,topics,emotions,audience,
                                         note_count,author_count,likes,views,fans,samples,week,created_by)
                   VALUES (%s,%s,%s,'ai',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (star_key) DO UPDATE SET title=EXCLUDED.title, layer=EXCLUDED.layer,
                     mechanism=EXCLUDED.mechanism, topics=EXCLUDED.topics, emotions=EXCLUDED.emotions,
                     audience=EXCLUDED.audience, note_count=EXCLUDED.note_count,
                     author_count=EXCLUDED.author_count, likes=EXCLUDED.likes, views=EXCLUDED.views,
                     fans=EXCLUDED.fans, samples=EXCLUDED.samples, week=EXCLUDED.week, updated_at=NOW()""",
                (key, title, rec["layer"], s.get("url", ""), mech,
                 json.dumps(rec["topics"], ensure_ascii=False), json.dumps(emo, ensure_ascii=False),
                 rec["audience"], rec["note_count"], rec["author_count"], rec["likes"], rec["views"],
                 rec["fans"], json.dumps(rec["samples"], ensure_ascii=False), week, u["email"]),
            )

        rels = payload.get("relations")
        if rels is None and payload.get("autolink", True):
            cur.execute(f"SELECT {_STAR_COLS} FROM neb_star")
            allstars = [_star_row(r) for r in cur.fetchall()]
            rels = nebula.infer_relations(allstars)
        for r in (rels or []):
            a, b = sorted([str(r["a"]), str(r["b"])])
            cur.execute(
                """INSERT INTO neb_rel (a,b,strength,kinds,reason) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (a,b) DO UPDATE SET strength=EXCLUDED.strength,
                     kinds=EXCLUDED.kinds, reason=EXCLUDED.reason""",
                (a, b, float(r.get("strength", 0.3)),
                 json.dumps(r.get("kinds") or [], ensure_ascii=False), r.get("reason", "")),
            )
        conn.commit()
    return {"ok": True, "ingested": len(norm), "relations": len(rels or [])}


@app.post("/api/nebula/from-paradigm")
def nebula_from_paradigm(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    """把范式雷达的聚类结果投射成星图（四区 → 恒星/星胚/伴星/尘埃）。"""
    u = _require_user(decrypted_userinfo)
    result = payload.get("result") or {}
    week = payload.get("week", "")
    bucket_layer = {"core": "core", "review": "review",
                    "burst_candidate": "motif", "burst_review": "motif", "excluded": "excl"}
    stars = []
    for bucket, layer in bucket_layer.items():
        for idx, c in enumerate(result.get(bucket, [])):
            key = f"p-{bucket}-{idx}-{abs(hash(c['name'])) % 100000}"
            stars.append({
                "id": key, "title": c["name"], "layer": layer, "mechanism": c.get("mechanism", ""),
                "topics": [t[0] for t in (c.get("top_topics") or [])],
                "note_count": c.get("note_count", 1), "author_count": c.get("author_count", 1),
                "likes": c.get("likes_median", 0),
                "views": int(c["likes_median"] / c["like_rate_median"]) if c.get("like_rate_median") else 0,
                "fans": 0,
                "samples": [n["title"] for n in (c.get("notes") or [])[:3]],
                "url": (c.get("top_note") or {}).get("url", ""),
            })
    if not stars:
        raise HTTPException(400, "范式雷达结果为空，先去「范式雷达」跑一次分析")
    return nebula_ingest({"week": week, "stars": stars, "autolink": True}, decrypted_userinfo)


@app.post("/api/nebula/relink")
def nebula_relink(decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo")) -> dict:
    _require_user(decrypted_userinfo)
    with _db() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {_STAR_COLS} FROM neb_star")
        stars = [_star_row(r) for r in cur.fetchall()]
        rels = nebula.infer_relations(stars)
        for r in rels:
            a, b = sorted([r["a"], r["b"]])
            cur.execute(
                """INSERT INTO neb_rel (a,b,strength,kinds,reason) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (a,b) DO UPDATE SET strength=EXCLUDED.strength,
                     kinds=EXCLUDED.kinds, reason=EXCLUDED.reason""",
                (a, b, r["strength"], json.dumps(r["kinds"], ensure_ascii=False), r["reason"]),
            )
        conn.commit()
    return {"ok": True, "relations": len(rels)}


@app.post("/api/nebula/report")
def nebula_report(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> dict:
    """点星星出小报告。AI 已配置走 Runway 网关；未配置返回结构化口径报告。"""
    _require_user(decrypted_userinfo)
    star = payload.get("star") or {}
    if not star.get("title"):
        raise HTTPException(400, "star 缺失")
    key = star.get("id", "")

    if not payload.get("force"):
        try:
            with _db() as conn, conn.cursor() as cur:
                cur.execute("SELECT report FROM neb_star WHERE star_key=%s", (key,))
                row = cur.fetchone()
                if row and row[0]:
                    return {"report": row[0], "cached": True}
        except HTTPException:
            pass

    ai = _load_ai_props()
    if not ai.get("ai.base_url") or not ai.get("ai.api_key"):
        return {"report": nebula.rule_report(star), "cached": False, "ai": False}

    import httpx
    try:
        resp = httpx.post(
            f"{ai['ai.base_url']}/bedrock_runtime/model/invoke",
            headers={"token": ai["ai.api_key"], "Content-Type": "application/json"},
            json={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": nebula.build_report_prompt(star)}],
            },
            timeout=60,
        )
        data = resp.json()
        if data.get("Code") or data.get("Error"):
            raise RuntimeError(f"AI call failed: {data}")
        rep = nebula.parse_report(data["content"][0]["text"])
        rep["_source"] = "ai"
    except Exception as e:
        rep = nebula.rule_report(star)
        rep["_note"] = f"AI 生成失败，已降级为口径报告：{e}"

    try:
        with _db() as conn, conn.cursor() as cur:
            cur.execute("UPDATE neb_star SET report=%s, updated_at=NOW() WHERE star_key=%s",
                        (json.dumps(rep, ensure_ascii=False), key))
            conn.commit()
    except HTTPException:
        pass
    return {"report": rep, "cached": False, "ai": rep.get("_source") == "ai"}


# ── 导出 ──

@app.post("/api/export")
def export_csv(
    payload: dict = Body(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> Response:
    _require_user(decrypted_userinfo)
    result = payload.get("result") or {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["区块", "Cluster名称", "内容机制", "笔记数", "独立作者数", "点赞中位数",
                "点赞率中位数", "破圈系数中位数", "标题一致性", "正文一致性", "话题一致性",
                "综合一致性", "复核状态/原因", "是否值得做工", "代表笔记", "代表笔记链接"])
    labels = {"core": "核心范式", "review": "人工复核Cluster", "burst_candidate": "有效万赞单点-范式候选",
              "burst_review": "需复核万赞单点", "excluded": "排除表"}
    for key, label in labels.items():
        for c in result.get(key, []):
            w.writerow([label, c["name"], c["mechanism"], c["note_count"], c["author_count"],
                        c["likes_median"], c["like_rate_median"], c["breakout_median"],
                        c["title_sim"], c["body_sim"], c["topic_sim"], c["overall_sim"],
                        c.get("review_reason", ""), "是" if c.get("worth_doing") else "否",
                        c["top_note"]["title"], c["top_note"]["url"]])
    return Response(
        content="\ufeff" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="paradigm_radar.csv"'},
    )
