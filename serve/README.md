# DS4 推理栈：API 封装、拓扑切换、KV 路由

本目录是 DS4（Maya-S，IQ2_XXS）的生产封装，方式与 `strata/` 栈一致：
同一个 `serve.server --engine strata` 前端、同一套 OpenAI/Anthropic 兼容端点、同一个 Web 界面。

## 1. 启动

```bash
# 选择拓扑（写 serve.json 软链并重启服务）
strata/ds4-z100l/serve/ds4-switch.sh tp4

# 或手动
systemctl start ds4-serve
```

- 服务单元：`/etc/systemd/system/ds4-serve.service`（`ExecStartPre` 建 KV 目录、`ExecStart` 起容器、
  `ExecStop` 优雅停止以便落盘 KV）；
- 端口：`127.0.0.1:18100`；
- 模型/引擎：`/work/maya-ds4-z100l/build-906/maya-ds4`（容器内）。

## 2. API 端点（OpenAI / Anthropic 双方言）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI 兼容，流式/非流式 |
| POST | `/v1/responses` | OpenAI Responses |
| POST | `/v1/messages` | Anthropic Messages |
| GET | `/v1/models` | 模型列表（健康检查用） |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/` | Web 界面（`serve/web/`，与 Maya 原版同一份） |

## 3. 并发：无并发、请求排队

- 引擎以 `--parallel 1` 运行：**同一时刻只服务一条会话**；
- 服务端用一把容量为 1 的锁串行化请求：后来的请求**按到达顺序排队**（`Lock` 的 FIFO 队列），
  不会并发跑两条流，也不会互相打断；
- 因此显存里**只需要一份 KV**（见第 4 节）。

## 4. KV 路由与冷存

引擎与前端把会话 KV 分成两层：

| 层 | 位置 | 机制 |
|---|---|---|
| 热 KV | 显存 | 当前会话一份，`--parallel 1` 保证只有一份 |
| 停泊 KV（parked） | 主机 RAM | 预算 `STRATA_DS4_SLOT_RAM_GB`（默认 8 GB），超出后**溢出到 SSD 槽位目录** |
| 冷存 KV | SSD `STRATA_DS4_SLOT_DIR` | 每会话一个槽位文件，容量 `STRATA_DS4_SLOT_GB`（GB/会话）与 `STRATA_DS4_SLOT_CAP`（会话数）封顶 |

- **退出/重启落盘**：`STRATA_DS4_SLOT_KEEP=1` 时，引擎停止前把当前会话写入槽位目录，
  下次启动可复用（systemd 的 `ExecStop` 用 `docker stop -t 120` 留足落盘时间）；
- **淘汰策略**：容量不足时按 **LRU**（使用次数少 + 最久未用优先）淘汰槽位文件；
- **与 strata 共用 SSD**：`STRATA_DS4_SLOT_DIR=/data/kv/ds4`，与 strata 的 KV 冷盘同一块 SSD
  （`/data/kv`），容量按目录配额分配，互不覆盖。

清空冷存：`rm -rf /data/kv/ds4/*`（服务停止时执行）。

## 5. 拓扑切换

| 拓扑 | 配置 | 实测（code 长任务，DSpark 开） | 备注 |
|---|---|---:|---|
| 单卡 | `serve-single.json` | ~13.7 tok/s | 专家大量走 CPU lane |
| TP2 | `serve-tp2.json` | ~12.9（CPU lane 74.7%） | 2 卡显存不足，不推荐 |
| TP3 | `serve-tp3.json` | ~35.3 tok/s | 需要 `DS4_TP_UNEVEN_HEADS=1` |
| **TP4** | `serve-tp4.json` | **~46.9 tok/s** | 推荐 |

切换只换配置、重启进程（同一份二进制），`ds4-switch.sh` 已封装。

## 6. 投机解码（DSpark）

- 所有配置默认 `--dspark on`，`DS4_DSPARK_MAX=8`；
- 短测（256 token）41–42 tok/s，长 code 任务（1200–1600 token）46.9 tok/s；
- `--suffix-draft 0`（prompt lookup 与 DSpark 叠加实测更差，故关闭）。

## 7. 三种实例：TP4 / TP3 / TP2

同一份引擎、同一份模型，三种拓扑各有一个配置；`serve.json` 是软链，`ds4-switch.sh` 或手工 `ln -sf` 切换后
`systemctl restart ds4-serve`。

| 实例 | 配置 | 卡 | `--tp` | `--max-context` | `--reserve-mib` | helper 池 |
|---|---|---|---|---|---|---|
| TP4（生产） | `serve-1m-memberhelper.json` | 0,1,2,3 | 4 | 1048576 | **1536** | `DS4_HELPER=1,2,3`，`DS4_HELPER_GB=10` |
| TP3 | `serve-tp3-mh.json` | 0,1,2 | 3 | 1048576 | **1536** | `DS4_HELPER=1,2`，`DS4_HELPER_GB=12` |
| TP2 | `serve-tp2-mh.json` | 0,1 | 2 | 1048576 | **4096** | `DS4_HELPER=1`，`DS4_HELPER_GB=12` |

三者都开了 `--dspark on`、`--prefill-chunk 16384`、`--mmap-experts auto`、`--parallel 1`、`DS4_TP_GGUF_UNEVEN=1`。

### 显存从哪来、给谁（实测）

```
lane_alloc            PCIe 通道的 staging（每卡固定）
  ↓
place_experts         TP 专家 arena = (free − reserve_mib) × vram_experts(1.0)
  ↓
helper_select         helper 池 = min(DS4_HELPER_GB, free − 768 MiB)   ← 最后，只吃剩下的
  ↓
load_experts          把专家搬进 arena / 池
```

**顺序不能变**：helper 池永远在 TP 专家之后分配，且夹紧到"当前空闲 − 768 MiB"。所以 TP 专家的位子先摆满，
helper 只拿剩饭 —— 这就是"绝不抢占已用显存"在代码里的落点（`ds4_model.cu` 的 `place_experts` → `helper_select`）。

`--reserve-mib` 是"每卡必须留空的 VRAM"（引擎默认 600，这里按拓扑给 1536/4096）。它覆盖：驱动与桌面、
启动之后才会发生的小分配（HIP code object、驱动队列）、以及碎片。实测在 1M 上下文 + 5 万 token 预填 +
72 请求的满负载下，它从未被真正用到，所以 2026-11-08 把 3072→1536、TP2 8192→4096 交给了 TP 专家 arena：

| 形态 | arena/卡（改前 → 改后） | 专家槽/卡 | 卡0 free | helper 卡 free |
|---|---|---|---|---|
| TP4 | 14.06 → **15.65 GB** | 5107 → **5687** | 3.18 → 1.57 GB | 0.81（池子缩到 ~8.4 GB） |
| TP3 | 15.56 → 待测 | 待测 | 待测 | 待测 |
| TP2 | 待测 | 待测 | 待测 | 待测 |

## 8. VRAM / OOM 监控器

`vram-guard.py`（本目录）+ `ds4-vram-guard.service`（systemd，`Restart=always`）：

```bash
systemctl status ds4-vram-guard          # 是否在跑
tail -f /var/log/ds4-vram-guard.log      # 告警（只在状态变化时写）
tail -f /data/build/vram-guard.jsonl     # 每 30 s 一行：每卡 free / min / state / slots / arena
```

它每 5 s 读一次引擎自己的 `/metrics`（不依赖 NVML，也不碰引擎），判定：

| 条件 | 级别 | 含义 |
|---|---|---|
| `free < 512 MiB` | WARN | reserve 被吃了：有东西在启动后分配显存 |
| `free < 192 MiB` | CRIT | 下一次分配就会失败（OOM 边缘） |
| `/metrics` 连续 6 次不可达 | CRIT | 引擎死了或卡住 |
| 容器日志出现 `no VRAM` / `out of memory` / `cudaMalloc` / `a launch failed` | CRIT | 已经发生分配失败 |
| `free < reserve − 768` | WARN | 破坏了"free 恒等于 reserve"的不变量 |

阈值可用环境变量改：`GUARD_WARN_MIB`、`GUARD_CRIT_MIB`、`GUARD_INTERVAL`、`GUARD_SAMPLE_S`、`GUARD_CONTAINER`。

**判读方法**：正常情况下 `free` 应该**恒等于** `--reserve-mib`（卡0）和 **768 MiB**（有 helper 池的卡），
一条直线。任何下滑都意味着"启动后分配"发生了 —— 那时先看 `GUARD` 告警时间点，再看引擎日志。

## 9. TP 专家池改建的实测结果（2026-11-08，时间加权 CPU 口径）

把 `--reserve-mib` 从 3072 降到 1536（TP2 从 8192 降到 4096），预留交给 TP 专家 arena；helper 池仍在
`place_experts` 之后按 `free − 768 MiB` 夹紧。三种形态各跑同一组 3 个 SWE-bench Lite 实例（真实负载），
全部 **3/3 resolved**：

| 形态 | prefill tok/s (P25/P95) | decode tok/s (P25/P95) | CPU 利用率 (P25/P95) | RAM 带宽 GB/s (P25/P95) | VRAM 命中 (P25/P95) |
|---|---|---|---|---|---|
| TP4 | 81.2 / 328.5 | 34.4 / 48.7 | 62.1 / 99.5 % | 6.7 / 14.5 | 0.86 / 0.89 |
| TP3 | 55.9 / 233.9 | 27.0 / 36.5 | 59.7 / 76.4 % | 11.1 / 17.7 | 0.82 / 0.87 |
| TP2 | 37.0 / 207.4 | 18.1 / 23.0 | 57.1 / 65.5 % | 13.9 / 19.7 | 0.61 / 0.67 |

CPU 利用率是**时间加权频率**：`used_ghz = ∫ f(t)·busy(t) dt / T`（每 0.5 s 采样、梯形积分，一个物理核取两个
SMT 线程忙时的并集），`share = Σ used_ghz / (16×3.6 GHz)`。满负载读数为 ~62 GHz = 107%（睿频 3.86 > 名义 3.6，
>100% 正常）。旧的「SMT 兄弟平均 × 瞬时 MHz」口径只剩 `cpu_pct_legacy` 作对照。

RAM 带宽是**真实 GB/s**：引擎 STAT 里的 `ram_fetch` 是"每 token 从内存读几个专家"（`cx/fed − disk`），
一个专家 11.0 MB（`ram_gb / ram_used`），所以 `带宽 = ram_fetch × 11.0 MB × tok_s`。之前把 `ram_fetch`
直接当 GB/s 用，17–90 其实是 experts/tok，真实带宽只有 6.7–19.7 GB/s —— 内存远没打满（上限 123 GB/s），
瓶颈在 CPU lane 的线程数与计算，不是带宽。旧键 `ram_fetch_gbs` 保留为 experts/tok 序列。

每卡状态（TP4/TP3）：卡0 free = 1536 MiB（正好是 reserve），helper 卡 free = 768 MiB，专家槽 5107 → 5687（TP4）。
TP2 的卡0 free = 4096 MiB（它的 reserve）。

## 10. 图像输入（已开启，2026-11-09）

**结论：DeepSeek-V4.1-Flash-1017 有原生视觉，之前没开是因为我们用的 GGUF 重打包把它去掉了。**

### 视觉塔在哪

* 任何 GGUF 量化包（包括 Maya-S）都**不含**视觉塔：6 个分片的 tensor 表里只有
  `blk.*.ffn_*_exps`、`blk.*.engram_*`、`token_embd`、`output*`，没有任何 `vision.*`/`visual.*`。
* 塔在 **DeepSeek 自己的 checkpoint 第 1 分片**：`deepseek-ai/DeepSeek-V4.1-Flash-1017`，
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`，`model-00001-of-00048.safetensors`
  （970,533,624 B，sha256 `13bd1210…d52fddb0ec`，只有 `vision.*` 32 块 + `aligner.*`，BF16）；
  几何在 `config.json` 的 `vision_config` + `image_token_id`（129264）；
  图片 span 的三个标记行（`image_start/newline/end`，各 10240 B）在**第 2 分片**里，用 30 KB range 请求取。
* 我们包的 tokenizer 本来就有 `<｜deepseek_image｜>`(129264)、`<｜rl_image_pad｜>`、`<｜rl_image_start｜>`，
  chat template 里也写着 `IMG = "<｜deepseek_image｜>"` —— 词表与模板都是视觉版的，缺的只是权重。

### 怎么装上的（可复现）

```bash
# 1. 取塔（0.97 GB，唯一的大下载）
R=1bae0e1088993b9b82de08f59de5be5a34e42170
curl -fL -o vision-src/config.json \
  "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/config.json"
curl -fL -o vision-src/model-00001-of-00048.safetensors \
  "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/model-00001-of-00048.safetensors"

# 2. 打成 vision.bin + vision.json（266 个 tensor，逐个与分片比对；标记行走 range 请求）
python tools/ds4/vision_pack.py --dir vision-src \
  --marks-url "https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/$R/model-00002-of-00048.safetensors"

# 3. 加进包（只追加，不动现有 tensor 布局）
python strata/ds4-z100l/serve/add-vision-to-pack.py --vision vision-src \
  --pack /models/deepseek-v41-flash-1017/patched/Maya-S/maya-ds4

# 4. 服务端要 Pillow + numpy 才能读图（venv 里）
/work/ds4-venv/bin/python -m pip install pillow numpy

# 5. 配置里 --vision on（五个 serve-*.json 都已设），重启
```

`vision.bin` 放在 `<model>/../vision/`，`tensors.txt` 里以 `file 7 ../vision/vision.bin` 引用（相对 `model_dir`）。

### 实测

| 项目 | 值 |
|---|---|
| 引擎自报 | `INFO vision=ds4`，32 块，每图最多 1024 tokens |
| 显存 | **两图之间不占**；读一张图时从**预填充缓冲借 ~318 MiB**，读完归还 |
| 直连引擎 | 图里写「51」→ 回答 **51**，7.6 s，prompt 241 tokens |
| 走聚合平台 | `LLM-server-heavy` → tp4 → 回答 **51**，5.5 s |
| `/health` | `images: true`（实例页「图像输入」显示「开」） |
| 三种拓扑 | TP4 **51**（7.6 s）、TP3 **51**（9.5 s）、TP2 **51**（12.9 s），都是同一张 360×180 的图、241 prompt tokens |

四个卡的空闲在两张图之后基本不变（卡0 1571 → 1596 MiB，其余 807–810 MiB）。

### 保留的坑

* 只有 **Maya-S/Maya-XS 这类 GGUF 重打包**缺塔；换成官方 HF 权重（含塔）就无需这一步，
  但显存/量化收益会没。
* 塔的权重在 RAM 里（pinned，~0.97 GB），不从 `DS4_WARM_CAP_GB` 里出，但会挤压 RAM 余量；
  低内存时注意。
* 图太小会被拒（`vision.min_pixels = 295936`）：测试用 360×180 能过，但太小的图会被放大到最小像素数。

## 11. 控制台与聚合平台

* `ds4-console.service`（:8090）：资源控制台（加速卡 / 整机 / 每核芯片 / 实例列表 + 预估 + 进度条）、
  每个实例一页（只切权重文件）、API 聚合平台页。
  资源数据**每 10 s 刷新一次**，取**前 10 s 的时间加权平均**（1 Hz 采样、梯形积分，按整 10 s 块切换），
  不是某一瞬间的读数；页面每 5 s 拉一次，块变了才重绘。
* `ds4-agg.service`（:18200）：对外唯一入口，OpenAI + Anthropic 双方言；三个模型名
  `LLM-server-heavy/-plus/-lite`，空映射逐级回落；四个发布地址（首尔 VPS / 广州 VPS / 局域网 / 本机）；
  token 总计与日统计（按本机时区切日，保留 400 天）、前缀缓存命中统计、每模型参数与「暴露给用户」开关；
  **分发 API**：给每个使用者单独发一把 `sk-z100l-…`，可停用 / 轮换 / 删除，用量按 key 分开统计。
* 聚合平台页：发布地址、分发 key 列表（复制 / 显示 / 停用 / 轮换 / 删除）、总计与今日卡片、
  近 30 天 tokens 堆叠柱状图、按模型名分布、模型映射与参数表；页面每 30 s 自刷新。
* `ds4-vram-guard.service`：每卡 free VRAM 守卫，`free < 512 MiB` 告警、`< 192 MiB` 判 OOM 边缘；
  重启/加载期不再误报。
* 接口文档：`strata/ds4-z100l/console/API.md`（控制台 API、聚合 API、新实例接入契约）。

## 12. 固定这个实例

```bash
# 二进制与镜像的身份（写进 PINNED.json，改动要一起改）
md5sum /data/build/maya-ds4-z100l/build-906/maya-ds4
docker inspect z100l-dev:latest --format '{{.Id}}'
# 生产配置（实体文件，不是软链）
ls -la /data/z100l-strata-port/strata/ds4-z100l/serve/serve.json
```

`serve.json` 由控制台的「开启」写入（实体文件），内容等于 `serve-1m-memberhelper.json` +
当前选择的权重 pack。`PINNED.json` 记录：镜像 id、引擎二进制 md5、配置名、reserve、helper 列表、
各形态 P25/P95、验证日期。
