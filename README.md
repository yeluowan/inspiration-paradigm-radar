# Inspiration Paradigm Radar / 灵感星图雷达

将分散的内容灵感、爆款笔记与用户情绪信号，整理为可分析的**内容范式**与可探索的**灵感星图**。

> 从内容样本中发现模式，再把模式变成可讨论、可复用的选题判断。

## 能力

- **范式雷达**：导入 CSV，基于标题、正文、话题与互动数据聚类内容范式。
- **灵感星图**：将范式、热点、个人灵感放在同一张关系图里，辅助关联发现。
- **可解释分析**：保留代表样本、聚类依据、互动指标与人工复核入口。
- **可选 AI 报告**：未配置 AI 时自动降级为规则化报告；不依赖任何专有模型。
- **本地优先**：不配置数据库也可完成 CSV 分析与体验内置 Demo；配置 PostgreSQL 后可保存数据集与星图快照。

## 快速开始

```bash
git clone https://github.com/<your-account>/inspiration-paradigm-radar.git
cd inspiration-paradigm-radar
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 3000
```

打开 `http://localhost:3000`。默认进入灵感星图；点击导航可体验内容范式雷达，并使用内置 Demo 或导入 CSV。

## CSV 字段

分析至少需要可映射为以下字段的数据：

| 字段 | 含义 |
|---|---|
| `title` | 标题 |
| `body` | 正文（可空） |
| `topics` | 话题，多个可用逗号分隔 |
| `likes` | 点赞数 |
| `views` | 浏览量（可空） |
| `author` | 作者标识（可空） |
| `url` | 样本链接（可空） |

## 可选配置

- `db.properties`：从 `db.properties.example` 复制并填写 PostgreSQL；用于保存数据集、快照与自定义星图。
- `ai.properties`：从 `ai.properties.example` 复制并填写兼容网关；用于生成扩展报告。未配置时不影响核心功能。
- `REQUIRE_SSO=true`：在内网部署时启用 SSO Header 校验；本地开发默认使用匿名 Demo 用户。

## 技术栈

FastAPI · PostgreSQL（可选）· 原生 HTML/CSS/JavaScript · 可解释规则聚类

## 隐私与数据

请只导入拥有处理权限的数据。仓库不包含业务数据、访问令牌、内部服务地址或任何用户隐私数据。

## License

[MIT](LICENSE)

## Public showcase

`public-showcase/` is a fully static, no-login visual demo. It can be deployed to GitHub Pages directly for sharing; it contains no business data or internal interfaces.
