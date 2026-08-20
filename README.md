# AgentRust

高性能 coding agent，Rust 实现，架构对齐 [pi](https://github.com/earendil-works/pi-mono)（`pi-ai` / `pi-agent-core` / `pi-coding-agent` 分层）。

## 目标

- 可嵌入：`--mode rpc` JSONL over stdin/stdout
- 可脚本化：`--mode json` 全事件输出、`-p` 一次性
- 高性能：增量上下文缓冲、prompt-cache 友好前缀、并行工具执行、零拷贝消息
- 会话：追加式 JSONL 树（id/parentId 分支），compaction 自包含检查点

完整设计见 [DESIGN.md](DESIGN.md)。

## 架构

| crate | 职责 | 对应 pi 包 |
|---|---|---|
| `agent-ai` | provider 适配、模型目录、凭据、SSE 流式、usage 记账 | `@earendil-works/pi-ai` |
| `agent-core` | LLM 循环、工具并行执行器、取消传播 | `@earendil-works/pi-agent-core` |
| `agent-session` | JSONL 会话树、上下文增量缓冲、compaction、事件总线、steer/followUp 队列 | `@earendil-works/pi-coding-agent` |
| `agent-cli` | rpc / json / print 前端 | pi 各运行模式 |

## 构建

```bash
cargo build --release       # 单二进制：target/release/agent.exe
cargo run -p agent-cli -- --help
```

## 运行

```bash
# RPC 模式（可嵌入）
agent rpc

# 一次性
agent print -p "Hello"

# 事件流
agent json
```

## 状态

- [x] M0 workspace 骨架（编译通过，零 warning）
- [ ] M1 agent-ai：Anthropic SSE + OpenAI-compatible 流式、usage 记账
- [ ] M2 agent-core：循环、并行工具、取消
- [ ] M3 agent-session：JSONL 树、增量上下文、compaction
- [ ] M4 rpc / json / print 前端
- [ ] M5 事件总线开放 + 插件
- [ ] M6 TUI（可选）

## License

MIT