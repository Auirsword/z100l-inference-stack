# Z100L 四卡推理栈

4 × Hygon Z100L（DCU，32 GiB/卡）上的 **DeepSeek-V4.1-Flash（Maya-S · IQ2_XXS）** 推理栈：
TP4 / TP3 / TP2 三种张量并行拓扑、helper 显存池、KV 冷存、VRAM 守卫，以及真实 SWE 负载下的实测数据。

这是**推理栈本身**——引擎补丁、服务配置、显存分配、监控与基准。控制台、API 聚合平台、知识库等不在这个仓库里。

## 硬件

| 项 | 值 |
|---|---|
| 加速卡 | 4 × Hygon Z100L（PCI `1d94:55b7`），每卡 32 GiB，HIP / ROCm |
| CPU | EPYC 7F52，16 核 / 32 线程，3.6 GHz 名义（睿频 3.86） |
| 内存 | 220 GB |
| 模型 | DeepSeek-V4.1-Flash-1017 · Maya-S · IQ2_XXS，约 302 GB，1M 上下文 |

## 目录

| 目录 | 内容 |
|---|---|
| `serve/` | 生产配置（`serve-1m-memberhelper.json` 等）、拓扑切换、VRAM 守卫、**完整说明见 [`serve/README.md`](serve/README.md)** |
| `patches/` | 引擎补丁（EP4、gfx906 采样、确定性、路由追踪…）与源码 sha256 |
| `native/` | 引擎原生侧（CMake + `src/` + `tests/`） |
| `ep/` | EP4 专家并行的接口与测试 |
| `p2p/` | 卡间 P2P / fused all-reduce |
| `tools/` | 基准与启停脚本（`ds4_bench.py`、`topo_bench.py`、`start_serve.py`、`switch_serve.py`…） |

## 快速开始

```bash
# 1. 引擎二进制：容器内 /work/maya-ds4-z100l/build-906/maya-ds4（由 patches/ 打在 Maya 源码上构建）
# 2. 选拓扑（写 serve.json 并重启服务）
serve/ds4-switch.sh tp4
# 3. 起服务
systemctl start ds4-serve
# 4. 验证
curl -s http://127.0.0.1:18100/v1/models \
     -H "Authorization: Bearer $(cat serve/API_KEY.txt)"
```

`serve/API_KEY.txt` 里是占位符 `sk-ds4-REPLACE_ME`，换成你自己的；`serve/serve.json` 里的 `api_key` 同步改。

## 三种拓扑

| 实例 | 卡 | `--tp` | 上下文 | `--reserve-mib` | helper 池 |
|---|---|---|---|---|---|
| TP4（生产） | 0,1,2,3 | 4 | 1048576 | 1536 | `DS4_HELPER=1,2,3`，`DS4_HELPER_GB=10` |
| TP3 | 0,1,2 | 3 | 1048576 | 1536 | `DS4_HELPER=1,2`，`DS4_HELPER_GB=12` |
| TP2 | 0,1 | 2 | 1048576 | 4096 | `DS4_HELPER=1`，`DS4_HELPER_GB=12` |

三者都开 `--dspark on`、`--prefill-chunk 16384`、`--mmap-experts auto`、`--parallel 1`、`DS4_TP_GGUF_UNEVEN=1`。

## 显存分配顺序（不能变）

```
lane_alloc      PCIe 通道 staging（每卡固定）
  ↓
place_experts   TP 专家 arena = (free − reserve_mib) × vram_experts
  ↓
helper_select   helper 池 = min(DS4_HELPER_GB, free − 768 MiB)   ← 最后，只吃剩下的
  ↓
load_experts    把专家搬进 arena / 池
```

helper 池永远在 TP 专家之后分配，并夹紧到"当前空闲 − 768 MiB"：先摆满 TP 专家，helper 只拿剩饭。
这就是"绝不抢占已用显存、也不浪费空余显存"在代码里的落点。

## 实测（真实 SWE-bench Lite 负载，三种形态各 3 个实例，全部 3/3 resolved）

| 形态 | prefill tok/s (P25/P95) | decode tok/s (P25/P95) | CPU 利用率 (P25/P95) | RAM 带宽 GB/s (P25/P95) | VRAM 命中 (P25/P95) |
|---|---|---|---|---|---|
| TP4 | 81.2 / 328.5 | 34.4 / 48.7 | 62.1 / 99.5 % | 6.7 / 14.5 | 0.86 / 0.89 |
| TP3 | 55.9 / 233.9 | 27.0 / 36.5 | 59.7 / 76.4 % | 11.1 / 17.7 | 0.82 / 0.87 |
| TP2 | 37.0 / 207.4 | 18.1 / 23.0 | 57.1 / 65.5 % | 13.9 / 19.7 | 0.61 / 0.67 |

## 三个容易搞错的量

1. **RAM 带宽是真实 GB/s。** 引擎 STAT 里的 `ram_fetch` 是"每 token 从内存读几个专家"（`cx/fed − disk`），
   一个专家 11.0 MB，所以 `带宽 = ram_fetch × 11.0 MB × tok_s`。直接把它当 GB/s 会高估约 4 倍
   （17–90 其实是 experts/tok，真实带宽只有 6.7–19.7 GB/s）。内存远没打满（上限 123 GB/s），
   瓶颈在 CPU lane 的线程数与计算。
2. **CPU 利用率是时间加权频率**：`used_ghz = ∫ f(t)·busy(t) dt / T`，`share = Σ used_ghz / (16×3.6 GHz)`。
   满负载约 62 GHz = 107%（睿频 3.86 > 名义 3.6，>100% 正常）。
3. **`--reserve-mib` 是每卡必须留空的显存**（TP4/TP3 1536，TP2 4096）。正常时卡 0 的 `free` 恒等于它，
   helper 卡的 `free` 恒等于 768 MiB；任何下滑都意味着"启动后分配"发生了。

## 更多

- 拓扑切换、KV 冷存、投机解码、视觉塔、VRAM 守卫、逐项实测：见 [`serve/README.md`](serve/README.md)
- 补丁清单与源码 sha256：`patches/`
- 基准脚本：`tools/ds4_bench.py`、`tools/topo_bench.py`

## 说明

- 路径（`/work/…`、`/models/…`、`/data/kv/…`）沿用生产机的布局，按自己的环境改。
- 引擎本体是 Maya（DeepSeek-V4.1-Flash 的推理实现），本仓库提供的是**在 Z100L 上把它跑成四卡生产服务**所需的
  补丁、配置、显存分配与验证方法。
