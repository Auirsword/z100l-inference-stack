# DS4 推理 API 使用手册

服务：`ds4-serve.service`（systemd 托管，容器内 `serve.server --engine strata`）
监听：`http://<主机IP>:18100`
模型名：`LLM-server-flash`
上下文：32768 tokens（实际可用 `--max-context` 可调，见配置）
并发：**单流**，所有请求按到达顺序排队（不会并发跑两条流）
认证：`Authorization: Bearer <API_KEY>` 或 `x-api-key: <API_KEY>`

> API_KEY 在 `strata/ds4-z100l/serve/serve-<拓扑>.json` 的 `"api_key"` 字段里。

## 1. 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI Chat Completions（支持流式） |
| POST | `/v1/responses` | OpenAI Responses API |
| POST | `/v1/messages` | Anthropic Messages API（支持流式） |
| GET | `/v1/models` | 模型列表与状态 |
| GET | `/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/` | Web 聊天界面（与 Maya/strata 同一份前端） |

## 2. curl 示例

```bash
export DS4_KEY="sk-ds4-xxxxxxxx"
export DS4_URL="http://192.168.0.123:18100"

# 非流式
curl -s "$DS4_URL/v1/chat/completions" \
  -H "Authorization: Bearer $DS4_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LLM-server-flash",
    "messages": [{"role": "user", "content": "用一句话解释 MoE 为什么受内存带宽限制"}],
    "temperature": 0,
    "max_tokens": 256
  }'

# 流式
curl -N "$DS4_URL/v1/chat/completions" \
  -H "Authorization: Bearer $DS4_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"LLM-server-flash","stream":true,
       "messages":[{"role":"user","content":"写一个 C++ 环形缓冲区"}],
       "max_tokens":512}'
```

## 3. Python（openai SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.0.123:18100/v1", api_key="sk-ds4-xxxxxxxx")
resp = client.chat.completions.create(
    model="LLM-server-flash",
    messages=[{"role": "user", "content": "写一个 Python 异步队列"}],
    temperature=0,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

> 回复里可能带 `reasoning_content`（思考内容），OpenAI SDK 会忽略它；
> 需要时从原始 JSON 里取。

## 4. Anthropic 方言

```bash
curl -s "$DS4_URL/v1/messages" \
  -H "x-api-key: $DS4_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"LLM-server-flash","max_tokens":256,
       "messages":[{"role":"user","content":"hello"}]}'
```

## 5. 重要限制（务必知道）

| 项 | 现状 |
|---|---|
| 并发 | **没有并发**：后到的请求排队，FIFO |
| 上下文 | 32768 tokens（可调大，但 prefill 时间线性增长） |
| 视觉 | **不支持**（当前模型 pack 只有文本） |
| 投机解码 | 默认开（DSpark，`DS4_DSPARK_MAX=8`） |
| 吞吐 | 短请求 ~29 tok/s；真实代码长任务 ~25 tok/s（4 根内存配置） |
| 首 token | 短 prompt <1 s；16K prompt ~40 s（预填） |
| KV | 会话停泊在 RAM（8 GB 预算），溢出到 `/data/kv/ds4`，LRU 淘汰 |

## 6. 运维

```bash
systemctl status ds4-serve        # 状态
systemctl restart ds4-serve       # 重启（会中断进行中的请求）
journalctl -u ds4-serve -f        # 日志
tail -f /data/build/serve/ds4-tp4-engine.log   # 引擎日志（含 tok/s、CPU lane）
curl -s http://127.0.0.1:18100/health
```

切换拓扑：`strata/ds4-z100l/serve/ds4-switch.sh tp4|tp3|tp2|single`

## 7. 错误约定

| HTTP | 含义 |
|---|---|
| 401 | 缺少/错误的 API key |
| 400 `context_length_exceeded` | prompt+max_tokens 超过上下文（**不会截断**） |
| 503 | 引擎正在启动/重启（引擎崩溃后自动重启，约 2–4 分钟） |
