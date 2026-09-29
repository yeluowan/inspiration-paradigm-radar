"""星图灵感雷达 — 星空数据层.

星体分层（与范式雷达四区一一对应）：
  core   恒星   核心范式（跨作者成立、可大规模复制）
  review 星胚   人工复核 Cluster（边界待定）
  motif  伴星   母题变体 / 万赞单点（高价值早期信号）
  excl   尘埃   排除表（保留供复盘）
  insp   灵感星 人工拖入：灵感 / 未过会选题 / 刷到的好范式

星座连线 REL：AI 或规则判定的星体间联系，strength 越高星座越亮。
"""

from __future__ import annotations

import json
import re
from typing import Any

# ─────────────────── 关系判定规则 ───────────────────
# 表面内容不同，但同选题 / 同用户情绪 / 同表达机制 → 连成星座

REL_KINDS = {
    "topic":     ("同选题", 0.34),
    "emotion":   ("同用户情绪", 0.30),
    "mechanism": ("同表达机制", 0.24),
    "audience":  ("同人群心智", 0.12),
}

EMOTION_LEXICON: dict[str, list[str]] = {
    "祛魅/反叛": ["祛魅", "反向", "拒绝", "不想", "别再", "翻车", "破防", "反差", "对抗", "互搏"],
    "松弛/治愈": ["松弛", "治愈", "慢下来", "躺平", "发呆", "无所事事", "randomly", "不赶路"],
    "被看见/共鸣": ["原来不止我", "我也是", "嘴替", "说出了", "共鸣", "陌生人", "普通人"],
    "掌控/成就": ["终于", "第一次", "做成了", "改造", "从0到", "攒了", "坚持"],
    "怀旧/情感联结": ["奶奶", "妈妈", "小时候", "老家", "旧", "回忆", "县城"],
    "窥私/好奇": ["蹲了", "跟拍", "偷偷", "真实的", "揭秘", "一天"],
}


def detect_emotion(text: str) -> list[str]:
    hits = []
    for emo, words in EMOTION_LEXICON.items():
        if any(w in text for w in words):
            hits.append(emo)
    return hits


def _tok(s: str) -> set[str]:
    s = re.sub(r"[\s，。、！？；：""''（）()\[\]【】~·—\-_/\\|+*#@%&$^<>{}.,!?:;\"']+", "", s or "")
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def infer_relations(stars: list[dict[str, Any]], min_strength: float = 0.28) -> list[dict[str, Any]]:
    """按规则判定星体间联系。返回 [{a,b,strength,kinds,reason}]。"""
    rels: list[dict[str, Any]] = []
    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            a, b = stars[i], stars[j]
            kinds, score, why = [], 0.0, []

            ta = {t.lstrip("#") for t in (a.get("topics") or [])}
            tb = {t.lstrip("#") for t in (b.get("topics") or [])}
            if ta and tb:
                ov = len(ta & tb) / min(len(ta), len(tb))
                if ov >= 0.5:
                    kinds.append("topic"); score += REL_KINDS["topic"][1] * ov
                    why.append(f"同选题：{'、'.join(list(ta & tb)[:2])}")

            ea = set(a.get("emotions") or [])
            eb = set(b.get("emotions") or [])
            if ea & eb:
                kinds.append("emotion"); score += REL_KINDS["emotion"][1]
                why.append(f"同用户情绪：{'、'.join(list(ea & eb)[:2])}")

            ma, mb = _tok(a.get("mechanism", "")), _tok(b.get("mechanism", ""))
            if ma and mb:
                ov = len(ma & mb) / min(len(ma), len(mb))
                if ov >= 0.32:
                    kinds.append("mechanism"); score += REL_KINDS["mechanism"][1] * ov
                    why.append("同表达机制")

            if a.get("audience") and a.get("audience") == b.get("audience"):
                kinds.append("audience"); score += REL_KINDS["audience"][1]
                why.append(f"同人群：{a['audience']}")

            if score >= min_strength:
                rels.append({
                    "a": a["id"], "b": b["id"],
                    "strength": round(min(1.0, score), 3),
                    "kinds": kinds, "reason": "；".join(why),
                })
    return rels


# ─────────────────── 打分 ───────────────────

def score_star(s: dict[str, Any]) -> dict[str, Any]:
    """0-100 运营价值分：数据表现 + 可复制性 + 早期性。"""
    likes, views, fans = s.get("likes", 0), s.get("views", 0), s.get("fans", 0)
    rate = (likes / views) if views else 0.0
    breakout = (views / fans) if fans else 0.0
    authors = s.get("author_count", 1)
    notes = s.get("note_count", 1)

    perf = min(40, rate / 0.15 * 40)                    # 点赞率 15% 满分
    reach = min(25, breakout / 8 * 25)                  # 破圈 8 倍满分
    repl = min(20, (authors - 1) / 4 * 20) if authors > 1 else (8 if notes == 1 else 0)
    early = 15 if (authors <= 2 and likes >= 10000) else max(0, 15 - (notes - 2) * 3)

    total = int(round(perf + reach + repl + early))
    return {
        "score": max(0, min(100, total)),
        "breakdown": {
            "数据表现": round(perf, 1), "破圈能力": round(reach, 1),
            "可复制性": round(repl, 1), "早期信号": round(early, 1),
        },
        "like_rate": round(rate, 4), "breakout": round(breakout, 2),
    }


# ─────────────────── 报告（AI 优先，无 AI 时给结构化口径报告）───────────────────

REPORT_PROMPT = """你是小红书内容策略分析师。基于以下一个内容范式的数据，写一份精炼的运营报告。

范式名称：{title}
分层：{layer_label}
内容机制：{mechanism}
笔记数：{note_count}　独立作者数：{author_count}
点赞中位数：{likes}　点赞率：{like_rate}　破圈系数：{breakout}
话题：{topics}
情绪标签：{emotions}
代表笔记标题：{samples}

严格按以下 5 段输出，每段 2-3 句，不要写标题以外的多余话术，不要用 markdown 标题符号：
为什么火|
用户心智|
可复制的表达框架|
运营切入点|
风险与边界|
"""

LAYER_LABEL = {"core": "恒星·核心范式", "review": "星胚·待复核", "motif": "伴星·母题变体/万赞单点",
               "excl": "尘埃·已排除", "insp": "灵感星·人工拖入"}


def build_report_prompt(s: dict[str, Any]) -> str:
    sc = score_star(s)
    return REPORT_PROMPT.format(
        title=s.get("title", ""), layer_label=LAYER_LABEL.get(s.get("layer"), s.get("layer", "")),
        mechanism=s.get("mechanism", "（未填写）"),
        note_count=s.get("note_count", 1), author_count=s.get("author_count", 1),
        likes=s.get("likes", 0), like_rate=f"{sc['like_rate']*100:.1f}%", breakout=sc["breakout"],
        topics="、".join(s.get("topics") or []) or "无",
        emotions="、".join(s.get("emotions") or []) or "无",
        samples="；".join((s.get("samples") or [])[:3]) or s.get("title", ""),
    )


def parse_report(text: str) -> dict[str, str]:
    keys = ["为什么火", "用户心智", "可复制的表达框架", "运营切入点", "风险与边界"]
    out: dict[str, str] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        matched = None
        for k in keys:
            if line.startswith(k):
                matched = k
                break
        if matched:
            cur = matched
            out[cur] = line[len(matched):].lstrip("|｜:： ").strip()
        elif cur:
            out[cur] = (out[cur] + " " + line).strip()
    return {k: out.get(k, "") for k in keys}


def rule_report(s: dict[str, Any]) -> dict[str, str]:
    """AI 未配置时的结构化口径报告（不编内容，只陈述已知数据）。"""
    sc = score_star(s)
    a, n = s.get("author_count", 1), s.get("note_count", 1)
    emo = "、".join(s.get("emotions") or []) or "未识别到明确情绪标签"
    return {
        "为什么火": f"点赞率 {sc['like_rate']*100:.1f}%、破圈系数 {sc['breakout']}"
                    f"（阅读/发布时粉丝数），说明流量主要来自推荐而非存量粉丝。"
                    + ("已有跨作者跟发，说明表达框架本身可被复用。" if a >= 2 else "目前仅单点信号，尚未验证跨作者可复制性。"),
        "用户心智": f"情绪标签：{emo}。话题落点：{('、'.join(s.get('topics') or []) or '未挂载话题')}。",
        "可复制的表达框架": s.get("mechanism") or "机制未填写，需人工补充（建议补 OCR/ASR 证据后再判定）。",
        "运营切入点": f"当前 {n} 篇 / {a} 位独立作者。"
                      + ("规模已成立，可直接进作者约稿池批量复制。" if a >= 3 else
                         "建议先小范围试跟发 2-3 位作者验证，再决定是否放量。"),
        "风险与边界": ("命中非范式语义规则，需人工复核后再投入。" if s.get("layer") == "review" else
                       "封面字幕与视频口播未纳入结构化字段，涉及此类信号需人工补 OCR/ASR 证据。"),
        "_source": "rule",
    }


# ─────────────────── DDL ───────────────────

DDL = """
CREATE TABLE IF NOT EXISTS neb_star (
  id SERIAL PRIMARY KEY,
  star_key TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  layer TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'manual',
  url TEXT,
  mechanism TEXT,
  topics JSONB DEFAULT '[]'::jsonb,
  emotions JSONB DEFAULT '[]'::jsonb,
  audience TEXT,
  note_count INTEGER DEFAULT 1,
  author_count INTEGER DEFAULT 1,
  likes BIGINT DEFAULT 0,
  views BIGINT DEFAULT 0,
  fans BIGINT DEFAULT 0,
  samples JSONB DEFAULT '[]'::jsonb,
  week TEXT,
  report JSONB,
  created_by TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_neb_star_layer ON neb_star(layer);
CREATE INDEX IF NOT EXISTS idx_neb_star_week ON neb_star(week);

CREATE TABLE IF NOT EXISTS neb_rel (
  id SERIAL PRIMARY KEY,
  a TEXT NOT NULL,
  b TEXT NOT NULL,
  strength REAL NOT NULL DEFAULT 0.3,
  kinds JSONB DEFAULT '[]'::jsonb,
  reason TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (a, b)
);
"""


# ─────────────────── Demo 星空 ───────────────────

def demo_graph() -> tuple[list[dict], list[dict]]:
    """15 颗聚类星：恒星5 / 星胚4 / 伴星1 / 尘埃5。"""
    raw = [
        # id, layer, 标题, 机制, 话题, 人群, 笔记数, 作者数, 赞, 阅读, 粉丝
        ("c0", "core", "离网感摄影师", "创作者以「离网感摄影师」身份进入陌生场域，随机拍陌生人并讲出对方的故事，槽位=拍摄地点/人群",
         ["人像", "陌生人", "县城"], "泛内容消费者", 3, 3, 19000, 88000, 9000),
        ("c1", "core", "游客照祛魅", "把标准游客照拍成反差版本，先立标准再打破，槽位=景点/反差方向",
         ["游客照", "反差", "旅行"], "年轻女性", 6, 5, 31000, 165000, 12000),
        ("c2", "core", "我好像拍到了⋯⋯", "以「我好像拍到了」开头制造悬念，正文揭晓一个日常里被忽略的瞬间，槽位=被拍对象",
         ["瞬间", "抓拍", "陌生人"], "泛内容消费者", 4, 4, 24000, 140000, 7000),
        ("c3", "core", "反向旅游", "刻意去没人知道的小城，不打卡不赶路，用低成本换松弛感，槽位=目的地",
         ["反向旅游", "小城", "旅行"], "都市白领", 5, 5, 22000, 130000, 6000),
        ("c4", "core", "左脑VS右脑互搏", "设定一个身体对抗任务，全程记录翻车过程，槽位=参与者身份",
         ["脑力挑战", "反差"], "泛内容消费者", 4, 4, 29000, 175000, 11000),

        ("c5", "review", "氛围感人像教程", "拆解3个姿势教用户拍出氛围感，附修图参数",
         ["摄影教程", "姿势"], "摄影爱好者", 3, 3, 14500, 92000, 22000),
        ("c6", "review", "夏日征集活动", "官方话题活动驱动的投稿合集",
         ["官方活动", "征集"], "全量用户", 4, 4, 10750, 68000, 480000),
        ("c7", "review", "手账拼贴教程", "材料清单+步骤图的标准教程结构",
         ["手工", "手账", "教程"], "学生", 3, 2, 9800, 61000, 15000),
        ("c8", "review", "城市漫步路线攻略", "以路线为骨架串联店铺与景点",
         ["城市漫步", "攻略", "路线"], "都市白领", 3, 3, 11200, 78000, 14000),

        ("c9", "motif", "生命力版游客照", "游客照祛魅的母题变体：用动态抓拍替代摆拍，强调「活着的感觉」",
         ["游客照", "反差", "生命力"], "年轻女性", 1, 1, 48000, 240000, 3000),

        ("c10", "excl", "县城观察日记（系列）", "单账号连载系列，未形成跨作者跟发",
         ["县城观察"], "泛内容消费者", 3, 1, 4200, 33000, 12000),
        ("c11", "excl", "相机新品开箱", "品牌/产品驱动，非表达框架",
         ["开箱", "测评"], "数码爱好者", 2, 2, 14000, 120000, 60000),
        ("c12", "excl", "某景区门票攻略", "单一目的地驱动",
         ["景点", "攻略"], "旅行人群", 2, 2, 5600, 48000, 9000),
        ("c13", "excl", "修图软件安利", "工具驱动",
         ["修图", "工具"], "摄影爱好者", 2, 2, 4800, 42000, 8000),
        ("c14", "excl", "明星同款穿搭", "IP/人物驱动",
         ["穿搭", "明星"], "年轻女性", 3, 3, 7300, 66000, 30000),
    ]
    stars = []
    for (sid, layer, title, mech, topics, aud, nc, ac, likes, views, fans) in raw:
        emo = detect_emotion(title + mech)
        stars.append({
            "id": sid, "layer": layer, "title": title, "mechanism": mech,
            "topics": topics, "emotions": emo, "audience": aud,
            "note_count": nc, "author_count": ac,
            "likes": likes, "views": views, "fans": fans,
            "source": "paradigm", "url": "", "week": "2026-W35",
            "samples": [title],
        })

    rels = infer_relations(stars)
    # 母题—变体强连接：伴星必须连回母题
    forced = [("c1", "c9", 0.92, ["motif"], "母题变体：生命力版是游客照祛魅的直接衍生"),
              ("c1", "c0", 0.58, ["emotion", "audience"], "同用户情绪：祛魅/反叛；同为「重新看见普通人」母题"),
              ("c1", "c2", 0.51, ["mechanism"], "同表达机制：先建立预期再打破"),
              ("c0", "c2", 0.46, ["topic", "mechanism"], "同选题：陌生人抓拍"),
              ("c3", "c1", 0.40, ["emotion"], "同用户情绪：对标准化体验的祛魅"),
              ("c5", "c7", 0.36, ["mechanism"], "同表达机制：教程式步骤结构")]
    have = {(r["a"], r["b"]) for r in rels} | {(r["b"], r["a"]) for r in rels}
    for a, b, s, k, why in forced:
        if (a, b) in have:
            for r in rels:
                if {r["a"], r["b"]} == {a, b}:
                    r.update({"strength": s, "kinds": k, "reason": why})
        else:
            rels.append({"a": a, "b": b, "strength": s, "kinds": k, "reason": why})
    return stars, rels


def demo_inspirations() -> list[dict]:
    return [{
        "id": "demo1",
        "title": "示例灵感星：拖一条链接或一句话进来",
        "desc": "选题会没过的想法、刷到的好范式、临时冒出来的念头，都可以拖进星空。"
                "填上它关联的范式，就会自动连成星座。",
        "url": "",
        "relatedIds": ["c0", "c1"],
        "demo": True,
    }]
