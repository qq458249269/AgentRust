# 高性能 Agent 实现方案（对齐 pi 架构）

目标：一个可嵌入、可脚本化（RPC）、可流式的高性能 coding agent。以 Rust + tokio 实现；目录名 AgentRust 即为此意。若改用 TS/其他语言，分层与热路径设计不变，只有并发/内存细节随语言落点不同。

pi 的架构结论（本方案的依据）：
- 功能四层切割：`ai`（provider/模型/凭据/流式）→ `agent-core`（LLM 循环/工具）→ `coding-agent`（会话/压缩/插件/编排）→ 薄前端（interactive/print/json/rpc 共用同一 runtime）
- 一切皆事件：`AgentSession` 向外推事件流，扩展与前端订阅同一总线
- 会话 = 追加式 JSONL 树（id/parentId），压缩条目是自包含检查点
- 工具并行执行（preflight 串行保证顺序，执行并发）
- 上下文重建按需、压缩在 `window - reserve` 阈值触发；性能关键在 prompt cache 命中与避免全量重建

---

## 1. 分层（crate 划分镜像 pi）

```
agent-ai        provider 适配、模型目录、凭据解析、SSE/流式、usage 记账
agent-core      Agent 状态机、工具执行器、取消传播（无 IO 概念，纯消息）
agent-session   会话 JSONL 树、上下文构建、压缩、事件总线、steer/followUp 队列、重试
agent-cli       rpc / json / print 三种前端（共用 runtime），TUI 另立可选 crate
```

依赖面只允许单向：core 依赖 ai，session 依赖 core，cli 依赖全部。session 不碰具体 provider 类型。

选型（现成 crate，不重复造轮子）：
| 需求 | crate |
|---|---|
| 异步运行时 | tokio |
| HTTP/2 + 连接池 + SSE | reqwest（features: http2, stream） |
| JSON/JSONL | serde + serde_json（手写增量 parse 用于大行） |
| 零拷贝大输出 | bytes（Bytes 贯穿消息内容） |
| 取消 | tokio_util::sync::CancellationToken（包装成 AbortSignal 语义） |
| 锁/集合 | parking_lot、dashmap（工具注册表） |
| 事件总线 | tokio::sync::broadcast（多订阅者） |
| CLI | clap |
| 日志 | tracing |
| 错误 | thiserror（库）+ 顶层 anyhow |

skipped: tiktoken-rs；先用 provider 返回的 usage 校准估算，误差大再加本地 tokenizer，改 `TokenEstimator` trait 即可。

## 2. 关键路径设计

### 2.1 Agent 主循环（agent-core）

```
loop {
  1. 组装上下文（见 §4）→ 请求 provider
  2. 流式消费：text/thinking delta → 推事件；toolCall 累积
  3. stopReason:
       tool_use  → preflight(串行) → 并发执行 → 结果回填 → 下一轮
       stop/error/length → 结束
  4. 每轮结束检查：溢出 → 压缩 → 重试（pi 的 overflow recovery）
}
```

状态机用显式 enum（Idle / Streaming / RunningTools / Compacting），避免隐式 flag 竞争。agent 层不直接持有锁：状态经通道流入 session 层，session 层单点持有 `AppendOnlyLog + 树索引`。

### 2.2 工具执行（并行 + 顺序承诺）

pi 语义照搬：同一 assistant 消息里的 tool call **preflight 串行**（保证 `tool_call` 拦截/修改按源顺序生效、session 状态同步），随后 **并发执行**；`tool_execution_end` 按完成顺序发，最终 toolResult 消息仍按源顺序落库。

Rust 实现：`JoinSet<(callId, result)>` + 完成顺序广播；落库前按源顺序重排。大输出：超过阈值（默认 2000 字符仅作上下文截断）截断进上下文，全文写临时文件并在结果里带 `fullOutputPath`（镜像 pi 的 BashExecutionMessage）。

### 2.3 消息队列（steer / followUp）

两队列 + 投递时机：
- steering：当前 assistant 轮的 tool call 执行完、下一次 LLM 调用前插入
- followUp：agent 完全 settle 后才投递

队列非空时主循环持续（`agent_settled` 事件 = 无队列静默态）。取消：Esc 一次只取消当前 turn，队列内容回编辑器/上抛给调用方，不吞消息。

### 2.4 取消传播

全局 `CancellationToken` 每个 turn 换一个。贯穿：本轮所有 in-flight 工具、provider 请求体。工具阻止取消（完成写盘）按 tool 类型标注 `cooperative`。压缩发生在取消后不再重试（`willRetry` 只在溢出恢复场景为 true）。

## 3. 会话存储：追加式 JSONL 树

格式（对齐 pi v3）：
```
{"type":"session","version":3,"id":uuid,"cwd":...}          // 首行
{"type":"message","id","parentId","timestamp","message":{role,content,usage}}
{"type":"compaction","id","parentId","summary","firstKeptEntryId","tokensBefore","retainedTail":[...]}
{"type":"branch_summary","id","parentId","fromId","summary"}
{"type":"custom","id","parentId","customType","data"}        // 不进 LLM 上下文
{"type":"label"|"model_change"|"thinking_level_change"|"session_info",...}
```

设计要点：
- **写入**：`BufWriter` + 异步批量 flush（每 turn 结束或 100ms 节流）；崩溃安全 = 只认完整行。不许全量重写（除物理压缩）。
- **读取**：内存持有 `Vec<Entry>` + `HashMap<id, idx>`；启动解析一次，之后全内存追加。物理压缩（去掉被 compaction 覆盖的旧条目）在满足条件（文件 > 阈值且压缩条目占比高）时后台做，先写临时文件再原子 rename。
- **树**：leaf 指针 + 逐级 parentId 回溯。`buildContextEntries` 从 leaf 到 root，遇到 compaction 条目则：
  - 有 `retainedTail` → 它是自包含检查点，之前整段忽略，之后从该条目继续
  - 否则回退 `firstKeptEntryId`（旧格式兼容）
- `/tree` 导航 = 换 leaf 指针 + 可选 branch summary（放弃分支压缩注入），不复制文件。

## 4. 上下文与 token 管理（性能主场）

pi 每轮 `buildSessionContext` 全量重建；Rust 版做增量，这是对 pi 的主要性能增益：

- **常驻上下文缓冲**：`SystemPrompt + 工具定义 + 历史消息` 作为前缀缓冲。普通轮次只 `push` 新消息，发送成本 O(new)。
- **仅三处全量重建**：session 打开、压缩后、分支切换后。缓冲内每条消息带 `ts` 与世代号（compaction 换代），世代不匹配即失效重建。
- **token 估算**：每条消息的估算与 last usage 缓存（pi 的 `getContextUsage` 同款思路：优先用最后 assistant 的 usage，尾巴用估算）。估算器按 provider 校准：`estimate = usage.input - sum(前序已估)` 反推差值，消偏差。
- **prompt cache 友好**（决定性性能因素）：
  - 前缀稳定序：system → 工具定义 → 会话历史，禁止在中间插入时间戳/随机字段
  - 模型选择处启用 provider 原生缓存（Anthropic `cache_control` 断点 / OpenAI/DeepSeek 自动前缀缓存），`PI_CACHE_RETENTION` 语义保留
  - 新轮次的请求体 = 上一轮体 + delta，delta 化组装后整体序列化
  - cache 命中时 R 计费与延迟都大幅下降；目标：长会话稳态命中率 > 90%
- **压缩（compaction）**：阈值 `ctxTokens > window - reserve(16k)`，`keepRecentTokens` 20k 从新往旧累计切点；cut 点只允许落在 user/assistant/tool_call/自定义条目上，绝不切 tool result。同构于 pi：
  - 压缩请求用**独立路由 session id + 关 prompt-cache 写**（一次性请求不值得写缓存）
  - 对话先串行化为文本 `[User]: ...[Assistant tool calls]: read(path=...)`，防止模型当对话续写
  - 工具结果句尾截断 2000 字符 + 截断标记
  - 累计文件跟踪：从被压缩消息与新近压缩条目的 details 里提取 readFiles/modifiedFiles，跨多次压缩累积
  - split turn（单轮超 keepRecentTokens）：历史摘要 + turn 前缀摘要合并
- **发送前压测**：超出窗口 → 立即压缩 → 重试当前轮（不丢用户意图）。

## 5. 并发模型

- **每 session 一个 actor**（`mpsc`），串行化所有状态变更；事件用 `broadcast` 出。前端、工具、RPC 客户端都不会碰共享锁。
- **工具并发** = actor 内派生的 `JoinSet`，结果回投。
- **网络**：全局 `reqwest::Client` 共享（HTTP/2 多路复用、连接池、TLS 会话复用），禁止每请求新建 client。流式读取用 `BytesStream`，delta 解析在单任务里，避免逐 token spawn。
- **背压**：broadcast 慢消费者丢事件由前端自行重放（事件幂等，落库为准）；RPC 输出用 `BufWriter` 按行 flush。

## 6. 事件模型（可扩展地基）

镜像 pi 事件集：`message_start/update/end`、`tool_execution_start/update/end`、`turn_start/end`、`agent_start/end/settled`、`compaction_start/end`、`queue_update`、`model_select`、`context`（可改消息）、`tool_call`（可拦截/改参）、`tool_result`（可改结果）、`session_*`。

- 状态与快照分离：事件携带增量（delta、toolCallId、usage），订阅方按需 `session.snapshot()` 取全量
- 插件 = 事件处理器注册 + 工具注册表 + 命令表。event bus 上每个事件按注册序执行，返回 `{block}|{modify}|{continue}`；热重载：`/reload` 语义 = 重建订阅表（会丢内存态，文档声明）

## 7. 重试与容错

- 网络错误：指数退避 + jitter；429 读 `retry-after`；5xx 重试上限 3
- 流中断（SSE 截断）：回退到重发最近完整消息前缀，缓存命中则代价近 0
- 溢出恢复：压缩 → 重试；`willRetry` 标志贯穿事件
- 工具崩溃：结果带 `isError`，不终止 agent（除非 `terminate` 标记）
- 会话写失败：内存态继续，标记 `dirty`，`session_shutdown` 时重试落盘

## 8. 前端（三合一 runtime）

- `--mode rpc`：LF 严格 framing（勿用按 Unicode 分行器），命令响应 + 异步事件同流；`id` 关联。嵌入方、IDE 插件主入口
- `--mode json`：事件全量 JSON 行，调试/管道
- `-p/--print`：一次性；stdin 并入首 prompt

TUI（ratatui，可选 crate `agent-tui`）：editor、消息视图、快捷键、通知；依赖 session 事件流，零业务耦合。先不做，RPC 已覆盖嵌入场景。

## 9. 里程碑

| M | 交付 | 验收 |
|---|---|---|
| M1 | agent-ai：Anthropic SSE + OpenAI compatible 流式、usage、凭据、模型目录（本地缓存） | 单请求流式打印、token/cost 记账准确 |
| M2 | agent-core：循环、并行工具、取消、截断落盘 | 多轮多工具并行完成；Ctrl+C 单轮可中止 |
| M3 | agent-session：JSONL 树、增量上下文、压缩、回写 | 长会话稳定压缩、崩溃可恢复、cache 命中率可观测 |
| M4 | rpc + json + print 前端 | `agent --mode rpc` 嵌入 demo（简单客户端跑通 steer/followUp） |
| M5 | 事件总线开放 + 工具/命令插件 | 外部工具注册、tool_call 拦截演示 |
| M6 | TUI（可选） | 交互可日常使用 |

每里程碑留一个 assert 自检测试（如：压缩后 `buildContext` token 数 < 阈值；并行工具结果按源顺序落库）。

## 10. 明确不做（YAGNI）

- 多 provider 全支持：v1 只 Anthropic + OpenAI-compatible，其余按 trait 扩展
- MCP：技能即 CLI 工具文档（pi 立场）
- 子 agent：v1 不做，用 RPC 起子进程或后续插件
- 计划模式/权限弹窗：授权放前端层/插件，不进 core
- 本地 tokenizer：先靠 usage 校准，误差不可接受再加