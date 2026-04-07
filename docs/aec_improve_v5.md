# AEC v1.12.0 改善紀錄

## 概述

基於 v1.11.0（PBFDAF subband + shadow + FDKF + RES + HPF + saturation detect + warmup mu + adaptive g_min）架構，新增 FDKF Two-stage Q + 3 項 bug fix + Shadow Q ratio。

---

## Benchmark 結果

### Farend Single-Talk ERLE + AECMOS (15 cases, AEC Challenge)

| 指標 | v1.11.0 | v1.12.0 | SpeexDSP | WebRTC AEC3 |
|------|---------|---------|----------|-------------|
| Mean ERLE | 10.6 dB | **11.7 dB** | 6.9 dB | 25.8 dB |
| AECMOS echo_mos | 2.76 | **3.20** | 3.09 | 4.47 |
| AECMOS deg_mos | 5.00 | **5.00** | 5.00 | 5.00 |

### Doubletalk (15 real cases)

| 指標 | v1.11.0 | v1.12.0 | SpeexDSP | WebRTC AEC3 |
|------|---------|---------|----------|-------------|
| Mean ERLE | — | **4.3 dB** | 1.3 dB | 3.7 dB |
| AECMOS echo_mos | — | **3.03** | 2.76 | 4.39 |
| AECMOS deg_mos | — | **3.52** | 3.90 | 2.51 |

### 改善重點

| 指標 | 變化 | 備註 |
|------|------|------|
| FS echo_mos | **+0.44** (2.76→3.20) | 自導入 FDKF 以來最大單次改善 |
| DT ERLE | **4.3 dB** vs AEC3 3.7 dB | 超越 AEC3 (+0.6 dB) |
| DT deg_mos | **3.52** vs AEC3 2.51 | 大幅領先 AEC3 (+1.01) |

---

## 改善項目

### 1. FDKF Two-stage Q — 主要改善項 (FS echo_mos +0.41)

**問題**：v1.11.0 的 FDKF 使用固定 Q=1e-5。低 Q → 低 P → 低 K → R 自我強化死鎖：

```
高 error → 高 R → 低 K → 更新慢 → 高 error（循環）
```

**方案**：Two-stage Q：
- `Q_high = 1e-3`：快速收斂階段。高 Q 維持 P 高 → K 在高 R 下仍保持合理 → 打破 R 死鎖
- `Q_low = 1e-5`：收斂後穩定追蹤
- 切換條件：連續 10 幀 ERLE > 6dB
- 偵測到收斂後，main 和 shadow filter 的 Q 同時從 Q_high 切換至 Q_low

**公式**：
```
K = P × |X|² / (|X|² × P + R)    # Kalman gain
W += K × E × conj(X)               # Weight update
P = (1 - K×X) × P + Q              # Covariance update

收斂前: Q = Q_high = 1e-3 → P 維持高 → K ≈ 0.5-0.9
收斂後: Q = Q_low  = 1e-5 → P 衰減    → K 慢速追蹤
```

### 2. 三項 Bug 修正

**Bug 1: Shadow filter 缺少 use_kalman**

Shadow filter 建構時未傳入 `use_kalman=True` → shadow 跑 NLMS，main 跑 FDKF。

修正：傳入 `use_kalman=self.config.use_kalman` 至 shadow 建構子。

**Bug 2: reset() 中 P/Q 不一致**

- P 被重設為 0.1（應為 0.5，與初始化一致）
- 收斂重設後 Q 未回到 Q_high

修正：`P.fill(0.5)`、`Q[:] = Q_high`。

**Bug 3: copy_weights_from() 缺少 P 複製**

FDKF 在 shadow→main 複製權重時需同步 P（error covariance）。缺少 P 複製會導致 main filter 的 P 與複製來的 W 不匹配 → Kalman gain 錯誤。

修正：在 `copy_weights_from()` 中新增條件式 P 複製。

### 3. Shadow Q Ratio (FS echo_mos +0.03)

**問題**：Bug 1 修正後，shadow 和 main 都使用 FDKF 且 Q 相同 → 收斂速度相同 → shadow_err ≈ main_err → copy gate（shadow_err < main_err × 0.7）永遠不觸發 → shadow 形同虛設。

**方案**：`shadow_q_ratio = 3.0`
- Shadow Q_high = main Q_high × 3.0 = 3e-3
- Shadow Q_low = main Q_low × 3.0 = 3e-5
- Shadow 收斂更快 → copy gate 觸發 → echo path 改變後幫助 main 更快恢復

新增 AecConfig 參數：`shadow_q_ratio: float = 3.0`

---

## 驗證方式

```bash
cd /path/to/AEC

# ERLE + PESQ 評測
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex

# AECMOS 評測
/opt/homebrew/bin/python3 python/eval_aecmos.py wav/aec_challenge/
```

---

## 後續方向

- FS echo_mos 與 AEC3 仍有 1.27 的差距（3.20 vs 4.47），主要受限於線性 filter ERLE 上限
- NN-based NLP 預期可進一步提升 echo_mos
- ~~C 版本需同步更新~~ (已同步)
