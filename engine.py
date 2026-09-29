"""内容范式雷达 — 聚类引擎.

严格实现方法论：
  一、前置：数据口径 + 常青词分层 + 处理原则
  二、聚类：三路特征(标题60/正文15/话题25) + 双通道成簇 + 范式语义判断 + 消费指标后置
  三、呈现：自然跟发Cluster(核心范式/人工复核) + 万赞单点观察池 + 排除表
  四、质量控制：合并同源、孤立点保留、字段校验

纯 Python 实现（无 sklearn/jieba），中文用字符 bi-gram 做特征，避免机械拆词。
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Any

# ─────────────────────────── 一、常青词分层 ───────────────────────────

DEFAULT_STOPWORDS: dict[str, list[str]] = {
    "平台泛词": ["小红书", "话题", "分享", "记录", "日常", "生活"],
    "摄影常青词": ["摄影", "拍照", "照片", "氛围感", "出片", "姿势", "技巧", "教程", "修图", "调色"],
    "手工绘画常青词": ["手工", "制作", "DIY", "教程", "材料", "成品", "绘画", "画画", "图纸", "步骤"],
    "旅游常青词": ["旅游", "旅行", "攻略", "景点", "打卡", "酒店", "民宿", "路线", "推荐", "城市"],
}

# 处理原则：常青词只降权不删除；且不机械拆词 —— 白名单里的词即使含常青词也保留全权重
DEFAULT_PROTECTED = ["摄影师", "旅行者", "手工人", "画师", "离网感", "反向旅游", "特种兵"]

EVERGREEN_WEIGHT = 0.15  # 常青词降权系数（不为 0，即"降权不删除"）

# 同义归一化：VS ≈ 对抗 ≈ 互搏
DEFAULT_SYNONYMS: dict[str, str] = {
    "vs": "对抗", "VS": "对抗", "pk": "对抗", "PK": "对抗", "互搏": "对抗", "battle": "对抗",
    "左右脑互搏": "左脑对抗右脑",
    "citywalk": "城市漫步", "CityWalk": "城市漫步", "city walk": "城市漫步",
    "特种兵旅游": "特种兵",
}

# ─────────────────────────── 范式语义判断规则 ───────────────────────────
# 以下不直接认定为范式（第二部分第3条）
NON_PARADIGM_RULES: list[tuple[str, str, list[str]]] = [
    ("platform_op", "平台运营/活动驱动", ["官方", "活动", "征集", "投稿", "薯队长", "报名", "抽奖", "话题活动", "挑战赛"]),
    ("tutorial", "攻略/教程/知识驱动", ["攻略", "教程", "怎么拍", "如何", "干货", "保姆级", "科普", "指南", "入门", "避坑指南"]),
    ("brand_ip", "品牌/IP/人物/产品驱动", ["联名", "品牌", "新品", "测评", "开箱", "代言", "旗舰", "官方账号", "同款购买"]),
    ("single_poi", "单一目的地/景点驱动", ["景区", "门票", "开放时间", "地址在", "位于", "游客中心"]),
    ("news", "新闻事件驱动", ["通报", "官宣", "发布会", "突发", "回应"]),
]

# ─────────────────────────── 文本处理 ───────────────────────────

_PUNCT = re.compile(r"[\s，。、！？；：""''（）()\[\]【】~·—\-_/\\|+*#@%&$^<>{}.,!?:;\"']+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def normalize(text: str, synonyms: dict[str, str]) -> str:
    t = (text or "").strip()
    for k in sorted(synonyms, key=len, reverse=True):
        if k and k in t:
            t = t.replace(k, synonyms[k])
    return t


def tokenize(text: str) -> list[str]:
    """中文按字符 bi-gram，英文/数字按整词。不做机械分词，避免拆坏叙事角色。"""
    text = normalize_case(text)
    toks: list[str] = []
    for seg in _PUNCT.split(text):
        if not seg:
            continue
        if _CJK.search(seg):
            if len(seg) <= 2:
                toks.append(seg)
            else:
                toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
                # 保留 3-gram 抓句式
                toks.extend(seg[i:i + 3] for i in range(len(seg) - 2))
        else:
            toks.append(seg.lower())
    return toks


def normalize_case(t: str) -> str:
    return (t or "").replace("\u3000", " ")


def build_weights(tokens: list[str], evergreen: set[str], protected: set[str]) -> dict[str, float]:
    """给每个 token 权重；命中常青词降权（不删除），受保护词保持全权重。"""
    w: dict[str, float] = defaultdict(float)
    for tok in tokens:
        if tok in protected or any(p in tok for p in protected):
            w[tok] += 1.0
            continue
        hit = tok in evergreen or any(e and e in tok for e in evergreen if len(e) >= 2)
        w[tok] += EVERGREEN_WEIGHT if hit else 1.0
    return dict(w)


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * big.get(k, 0.0) for k, v in small.items())
    if dot <= 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def inclusion(a: set[str], b: set[str]) -> float:
    """包含度：避免核心话题被额外挂词稀释。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ─────────────────────────── 主流程 ───────────────────────────

def analyze(
    notes: list[dict[str, Any]],
    *,
    min_likes: int = 1000,
    burst_likes: int = 10000,
    sim_threshold: float = 0.22,
    like_rate_gate: float = 0.10,
    breakout_gate: float = 3.0,
    stopwords: dict[str, list[str]] | None = None,
    protected: list[str] | None = None,
    synonyms: dict[str, str] | None = None,
) -> dict[str, Any]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    protected_set = set(protected or DEFAULT_PROTECTED)
    syn = synonyms or DEFAULT_SYNONYMS
    evergreen: set[str] = {w for lst in stopwords.values() for w in lst}

    # ── 前置：口径过滤（仅点赞门槛，点赞率/破圈系数后置）──
    clean, dropped = [], []
    seen_ids: set[str] = set()
    for raw in notes:
        n = _coerce(raw)
        if not n:
            dropped.append({"reason": "字段缺失/无效笔记", "raw": str(raw)[:120]})
            continue
        if n["note_id"] in seen_ids:
            dropped.append({"reason": "笔记ID重复", "raw": n["note_id"]})
            continue
        seen_ids.add(n["note_id"])
        if n["likes"] < min_likes:
            dropped.append({"reason": f"未达候选池点赞门槛({min_likes})", "raw": n["note_id"]})
            continue
        clean.append(n)

    # ── 三路特征 ──
    for n in clean:
        title_n = normalize(n["title"], syn)
        body_n = normalize(n["body"], syn)
        n["_t"] = build_weights(tokenize(title_n), evergreen, protected_set)
        n["_b"] = build_weights(tokenize(body_n), evergreen, protected_set)
        n["_p"] = {normalize(t, syn).lower().lstrip("#") for t in n["topics"] if t}

    # ── 相似度 + 连通分量聚类（先算法后语义）──
    N = len(clean)
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    edges: list[tuple[int, int, float]] = []
    for i in range(N):
        for j in range(i + 1, N):
            st = cosine(clean[i]["_t"], clean[j]["_t"])
            sb = cosine(clean[i]["_b"], clean[j]["_b"])
            sp = inclusion(clean[i]["_p"], clean[j]["_p"])
            sim = 0.60 * st + 0.15 * sb + 0.25 * sp
            if sim >= sim_threshold:
                union(i, j)
                edges.append((i, j, sim))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)

    edge_map: dict[tuple[int, int], float] = {(i, j): s for i, j, s in edges}

    core, review, burst_candidate, burst_review, excluded, isolated = [], [], [], [], [], []

    for members in groups.values():
        items = [clean[i] for i in members]
        authors = {it["author_id"] for it in items}
        top = max(items, key=lambda x: x["likes"])

        cl = _build_cluster(items, members, edge_map, clean, evergreen, protected_set)
        flags = _semantic_flags(items)
        cl["flags"] = flags

        # 双通道成簇
        natural = len(items) >= 2 and len(authors) >= 2
        single_burst = len(items) == 1 and top["likes"] >= burst_likes

        if natural:
            cl["channel"] = "自然跟发Cluster"
            # 系列号抑制：同作者系列只算一个独立作者信号
            if len(authors) < 2:
                cl["channel"] = "孤立点"
            if flags:
                cl["review_reason"] = "；".join(f[1] for f in flags)
                review.append(cl)
            else:
                core.append(cl)
        elif single_burst:
            cl["channel"] = "万赞单点信号"
            if flags:
                cl["review_reason"] = "；".join(f[1] for f in flags)
                burst_review.append(cl)
            else:
                burst_candidate.append(cl)
        else:
            if len(items) >= 2:
                cl["review_reason"] = "独立作者不足2位（疑似同作者系列）"
                excluded.append(cl)
            else:
                cl["review_reason"] = "非万赞且未形成跨作者跟发"
                isolated.append(cl)

    # 消费指标后置：只评价不过滤
    for bucket in (core, review, burst_candidate, burst_review, excluded, isolated):
        for cl in bucket:
            cl["worth_doing"] = (
                cl["like_rate_median"] >= like_rate_gate and cl["breakout_median"] >= breakout_gate
            )

    core.sort(key=lambda c: (-c["note_count"], -c["likes_median"]))
    review.sort(key=lambda c: -c["likes_median"])
    burst_candidate.sort(key=lambda c: -c["likes_median"])
    burst_review.sort(key=lambda c: -c["likes_median"])
    excluded.sort(key=lambda c: -c["likes_median"])

    covered = sum(c["note_count"] for c in core + review + burst_candidate + burst_review)

    return {
        "params": {
            "min_likes": min_likes, "burst_likes": burst_likes,
            "sim_threshold": sim_threshold,
            "like_rate_gate": like_rate_gate, "breakout_gate": breakout_gate,
        },
        "overview": {
            "input_notes": len(notes),
            "valid_notes": len(clean),
            "dropped": len(dropped),
            "core_clusters": len(core),
            "review_clusters": len(review),
            "burst_candidates": len(burst_candidate),
            "burst_reviews": len(burst_review),
            "excluded_clusters": len(excluded),
            "isolated_notes": sum(c["note_count"] for c in isolated),
            "coverage": round(covered / len(clean), 4) if clean else 0.0,
            "worth_doing": sum(1 for c in core if c["worth_doing"]),
            "authors": len({n["author_id"] for n in clean}),
        },
        "core": core,
        "review": review,
        "burst_candidate": burst_candidate,
        "burst_review": burst_review,
        "excluded": excluded,
        "dropped": dropped[:200],
        "stopwords": stopwords,
    }


def _coerce(raw: dict[str, Any]) -> dict[str, Any] | None:
    def g(*keys, default=""):
        for k in keys:
            if k in raw and raw[k] not in (None, ""):
                return raw[k]
        return default

    nid = str(g("note_id", "笔记ID", "id", "noteId")).strip()
    title = str(g("title", "标题")).strip()
    if not nid or not title:
        return None
    topics_raw = g("topics", "话题", default="")
    if isinstance(topics_raw, str):
        topics = [t.strip() for t in re.split(r"[,，;；#\s]+", topics_raw) if t.strip()]
    else:
        topics = [str(t).strip() for t in topics_raw if str(t).strip()]

    def num(*keys, default=0):
        v = g(*keys, default=default)
        try:
            return int(float(str(v).replace(",", "").replace("w", "0000")))
        except Exception:
            return default

    return {
        "note_id": nid,
        "title": title,
        "body": str(g("body", "正文", "desc", default="")),
        "topics": topics,
        "author_id": str(g("author_id", "作者ID", "author", default="unknown")),
        "publish_time": str(g("publish_time", "发布时间", default="")),
        "category": str(g("category", "类目", default="")),
        "likes": num("likes", "点赞", "点赞数"),
        "views": num("views", "阅读", "阅读数"),
        "fans": num("fans", "作者粉丝数", "粉丝数"),
        "url": str(g("url", "笔记链接", "link", default="")),
    }


def _build_cluster(items, members, edge_map, clean, evergreen, protected) -> dict[str, Any]:
    likes = [i["likes"] for i in items]
    rates = [(i["likes"] / i["views"]) if i["views"] else 0.0 for i in items]
    breaks = [(i["views"] / i["fans"]) if i["fans"] else 0.0 for i in items]

    # cluster 内一致性：三路分别取成员两两均值
    ts, bs, ps = [], [], []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            i, j = members[a], members[b]
            ts.append(cosine(clean[i]["_t"], clean[j]["_t"]))
            bs.append(cosine(clean[i]["_b"], clean[j]["_b"]))
            ps.append(inclusion(clean[i]["_p"], clean[j]["_p"]))
    avg = lambda x: round(sum(x) / len(x), 3) if x else 1.0

    name, mechanism, slots = _name_cluster(items, evergreen, protected)
    top = max(items, key=lambda x: x["likes"])

    return {
        "name": name,
        "mechanism": mechanism,
        "slots": slots,
        "note_count": len(items),
        "author_count": len({i["author_id"] for i in items}),
        "likes_median": int(statistics.median(likes)),
        "likes_sum": sum(likes),
        "like_rate_median": round(statistics.median(rates), 4),
        "breakout_median": round(statistics.median(breaks), 2),
        "title_sim": avg(ts), "body_sim": avg(bs), "topic_sim": avg(ps),
        "overall_sim": round(0.6 * avg(ts) + 0.15 * avg(bs) + 0.25 * avg(ps), 3),
        "top_note": {"title": top["title"], "url": top["url"], "likes": top["likes"], "author_id": top["author_id"]},
        "notes": sorted(
            [{"note_id": i["note_id"], "title": i["title"], "author_id": i["author_id"],
              "likes": i["likes"], "views": i["views"], "fans": i["fans"], "url": i["url"],
              "like_rate": round((i["likes"] / i["views"]) if i["views"] else 0, 4),
              "breakout": round((i["views"] / i["fans"]) if i["fans"] else 0, 2),
              "publish_time": i["publish_time"], "topics": i["topics"]}
             for i in items], key=lambda x: -x["likes"]),
        "top_topics": _top_topics(items),
        "review_reason": "",
        "channel": "",
    }


def _common_phrase(titles: list[str], evergreen: set[str], protected: set[str]) -> str:
    """取跨标题最长的高覆盖公共短语作为 Cluster 锚点（常青词不能单独成锚）。"""
    if len(titles) == 1:
        return titles[0][:14]
    need = max(2, (len(titles) + 1) // 2)
    base = titles[0]
    for ln in range(min(12, len(base)), 1, -1):
        for s in range(0, len(base) - ln + 1):
            cand = base[s:s + ln].strip()
            if len(cand) < 2 or _PUNCT.fullmatch(cand):
                continue
            if sum(1 for t in titles if cand in t) < need:
                continue
            protected_hit = any(p in cand for p in protected)
            if not protected_hit and (cand in evergreen or (len(cand) <= 3 and any(e == cand for e in evergreen))):
                continue
            return cand
    return base[:12]


def _name_cluster(items, evergreen, protected) -> tuple[str, str, list[str]]:
    """从标题公共短语提取 Cluster 名 + 可替换槽位。"""
    titles = [i["title"] for i in items]
    anchor = (_common_phrase(titles, evergreen, protected) or titles[0][:12]).strip("，。、！？；：,.!?;: 　")
    name = f"{anchor} · 范式" if len(items) > 1 else f"{anchor}（万赞单点）"

    # 可替换槽位：成员标题里差异最大的成分
    slots = []
    heads = [re.split(r"[，,：:｜|/\s]", i["title"])[0] for i in items]
    if len(set(heads)) > 1:
        slots.append("主体/对象（标题首段各不相同）")
    if len({tuple(sorted(i["topics"])) for i in items}) > 1:
        slots.append("场景话题")
    if len({i["category"] for i in items}) > 1:
        slots.append("垂类")
    mechanism = (
        f"共享表达框架「{anchor}」，{len(items)} 篇 / {len({i['author_id'] for i in items})} 位独立作者跟发"
        + ("；可替换槽位：" + "、".join(slots) if slots else "；槽位待人工确认")
    )
    return name, mechanism, slots


def _top_topics(items) -> list[list[Any]]:
    c: dict[str, int] = defaultdict(int)
    for i in items:
        for t in i["topics"]:
            c[t.lstrip("#")] += 1
    return [[k, v] for k, v in sorted(c.items(), key=lambda x: -x[1])[:8]]


def _semantic_flags(items) -> list[tuple[str, str]]:
    """范式语义判断：命中则不直接认定为范式，转人工复核。"""
    text = " ".join((i["title"] + " " + i["body"] + " " + " ".join(i["topics"])) for i in items)
    flags = []
    for key, label, words in NON_PARADIGM_RULES:
        hits = [w for w in words if w in text]
        if len(hits) >= 2 or (hits and len(items) <= 2):
            flags.append((key, f"{label}（命中：{'/'.join(hits[:3])}）"))
    # 同一句文案复用但无槽位
    titles = [i["title"] for i in items]
    if len(titles) > 1 and len(set(titles)) == 1:
        flags.append(("no_slot", "同一句文案复用但没有可替换槽位"))
    # 单账号系列
    if len({i["author_id"] for i in items}) == 1 and len(items) > 1:
        flags.append(("single_account", "单账号系列内容"))
    return flags


# ─────────────────────────── Demo 数据 ───────────────────────────

def demo_notes() -> list[dict[str, Any]]:
    rows = [
        # 范式 A：左脑VS右脑互搏（跨作者跟发）
        ("n001", "左脑VS右脑互搏，我输得很彻底", "试了一下同时画圆和方，全程笑场，最后手不听使唤", "脑力挑战,反差", "a1", 42000, 210000, 12000, "娱乐"),
        ("n002", "左脑对抗右脑，程序员版本挑战失败", "写代码的手做不了这个测试，第三次就崩了", "脑力挑战,程序员", "a2", 31000, 180000, 8000, "娱乐"),
        ("n003", "左脑PK右脑，美术生也翻车了", "以为自己稳赢，结果比普通人还差", "脑力挑战,美术生", "a3", 27000, 150000, 20000, "娱乐"),
        ("n004", "左脑VS右脑，和我妈一起玩笑到缺氧", "家庭版互搏，我妈赢了我", "脑力挑战,家庭", "a4", 18000, 96000, 5000, "娱乐"),
        # 范式 B：离网感摄影师（保护词，不能被"摄影"降权拆掉）
        ("n011", "在县城当离网感摄影师，一天拍了37个陌生人", "带着胶片机在老街走，遇到谁拍谁，讲了很多故事", "人像,县城", "b1", 26000, 120000, 9000, "摄影"),
        ("n012", "离网感摄影师在菜市场蹲了一整天", "摊主一开始警惕，后来主动摆姿势", "人像,菜市场", "b2", 19000, 88000, 4000, "摄影"),
        ("n013", "做了三个月离网感摄影师，攒了800张陌生人肖像", "没有修图没有摆拍，就是记录", "人像,肖像", "b3", 15000, 70000, 30000, "摄影"),
        # 范式 C：反向旅游（跨作者）
        ("n021", "反向旅游第7站：去没人知道的小城待三天", "不打卡不赶路，就在县城菜场和公园坐着", "反向旅游,小城", "c1", 22000, 130000, 6000, "旅游"),
        ("n022", "反向旅游，我在鹤岗住了一周", "房租一个月300，生活成本低到不真实", "反向旅游,鹤岗", "c2", 35000, 200000, 11000, "旅游"),
        ("n023", "反向旅游合集，五个没人去的县城", "都是随机买票去的，反而更松弛", "反向旅游,县城", "c3", 12000, 90000, 3000, "旅游"),
        # 教程类（会被语义规则拦到人工复核）
        ("n031", "人像摄影教程：3个姿势拍出氛围感", "保姆级教程，新手也能学会，附修图参数", "摄影教程,姿势", "d1", 16000, 100000, 25000, "摄影"),
        ("n032", "摄影教程｜如何拍出氛围感人像，干货攻略", "入门指南，从构图到调色一步步教", "摄影教程,构图", "d2", 13000, 82000, 18000, "摄影"),
        # 平台活动驱动
        ("n041", "官方活动｜薯队长带你投稿夏日征集", "报名参加活动就有机会获得奖励，挑战赛开启", "官方活动,征集", "e1", 11000, 70000, 500000, "平台"),
        ("n042", "夏日征集活动投稿第二弹，官方话题活动", "参与活动抽奖，报名入口在评论区", "官方活动,投稿", "e2", 10500, 65000, 480000, "平台"),
        # 万赞单点（范式候选）
        ("n051", "把奶奶的旧衣服改成了一整套婴儿服", "拆了三件旧棉衣，缝了两周，妈妈看哭了", "改造,家庭", "f1", 48000, 240000, 3000, "手工"),
        # 万赞单点（需复核-品牌驱动）
        ("n052", "XX相机新品开箱测评，联名限定款", "官方寄的样机，测评一下画质和对焦", "开箱,测评", "g1", 14000, 120000, 60000, "数码"),
        # 单账号系列
        ("n061", "县城观察日记01：早市的六点半", "系列第一集", "县城观察", "h1", 5000, 40000, 12000, "生活"),
        ("n062", "县城观察日记02：修鞋摊的老王", "系列第二集", "县城观察", "h1", 4200, 33000, 12000, "生活"),
        ("n063", "县城观察日记03：录像厅还没关门", "系列第三集", "县城观察", "h1", 3800, 30000, 12000, "生活"),
        # 孤立点
        ("n071", "在阳台种活了一棵柠檬树", "养了两年终于结果", "种植", "i1", 2200, 26000, 1500, "生活"),
        ("n072", "手绘一整本城市地图送给爸爸", "画了60小时", "绘画,礼物", "j1", 6400, 52000, 2000, "手工"),
    ]
    return [
        {"note_id": r[0], "title": r[1], "body": r[2], "topics": r[3], "author_id": r[4],
         "likes": r[5], "views": r[6], "fans": r[7], "category": r[8],
         "publish_time": "2026-08-%02d" % (10 + (idx % 18)),
         "url": f"https://www.xiaohongshu.com/explore/{r[0]}"}
        for idx, r in enumerate(rows)
    ]
