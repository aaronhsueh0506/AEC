# Bisect analysis + plan priority (2026-04-22)

## 800-case 6-run bisect table

| Run         | FS_echo | DT_echo | DT_deg | NE_deg | ΔFS    | ΔDT_echo |
|-------------|---------|---------|--------|--------|--------|----------|
| Baseline    | 3.621   | 4.115   | 2.386  | 4.003  | —      | —        |
| full        | 3.474   | 4.034   | 2.445  | 4.005  | −0.147 | −0.081   |
| bisect_a    | 3.467   | 4.032   | 2.451  | 4.006  | −0.154 | −0.083   |
| bisect_b3a  | 3.486   | 4.043   | 2.431  | 4.004  | −0.135 | −0.072   |
| bisect_b3b  | 3.472   | 4.034   | 2.447  | 4.006  | −0.149 | −0.081   |
| bisect_b3c  | 3.477   | 4.036   | 2.449  | 4.005  | −0.144 | −0.079   |
| bisect_b7   | 3.475   | 4.035   | 2.448  | 4.006  | −0.146 | −0.080   |
| bisect_b11  | 3.474   | 4.028   | 2.447  | 4.004  | −0.147 | −0.087   |

### 結論

- 沒有任何單一 flag 對 800-case 有顯著貢獻 (|ΔFS| < 0.012)
- **B-3a** 最大單貢獻者 (+0.012 ΔFS)，但同時 DT_deg −0.014 (Pareto trade-off)
- **Fix A** 實際有正貢獻 (A OFF → FS 更差 0.007)
- **B-11** ON 救了 DT_echo (B-11 OFF → DT_echo −0.006)
- 整體 −0.147 退步約 92% 來自非 flag 控制的 unconditional 改動
- **不 revert 任何 flag** — 每個都有正貢獻或 Pareto trade-off

## Worst FS-gap top-20 分類

來源: `python/output_v25/worst_fs_gap.csv`

| Category | Count | 說明 |
|----------|-------|------|
| A (echo onset, FS no-movement) | **12** | raw_dt bug (B-15 候選) |
| B (saturation) | 0 | — |
| C (DT 誤判) | 0 | — |
| D (movement) | 8 | EPC / delay tracking |
| E (other) | 0 | — |

**全部 20 case 都是 `farend_singletalk`**。gap 範圍 1.56–2.33 dB (vs aec2 baseline)。

### Top-20 清單

Category A (12):
- JteZUZ4JYkeD4k2rcVbqHg (gap 2.33)
- VGlWeOPC6UiXSq4SYPiKpw (2.28)
- JLNgGcvTNEqbTDbc28wLkg (2.11)
- VJfVUwJs4k25ziMNvJb43A (2.00)
- r7U6JmcRl0ibIh0mN3CP9g (1.81)
- 9xjhiFbGo06hdQIsHTS6qA (1.81)
- lV0kQN0hR0ySmE0bQhuYbw (1.74)
- sLWe8bfYbkGwX1W3PzI1PQ (1.69)
- wr54weKzNkOcZ07hB04kzA (1.67)
- IxgmaPghzUGnR6sxrbGU3Q (1.62)
- s0oJqM6Y1UCHSVmHmgsx4Q (1.59)
- HIMqDWjSoECJFtIP0TM9bg (1.56)

Category D (8):
- iOyPaxX11UOaUkcscKhq1A_with_movement (2.27)
- s0oJqM6Y1UCHSVmHmgsx4Q_with_movement (2.07)
- JjCzlhn3gEiBQvfJtPNJ9A_with_movement (1.80)
- kHsrUmyfT0O0RYtusGuQyQ_with_movement (1.74)
- Ja8OngfthkOCmL8ldcRNyg_with_movement (1.64)
- VJfVUwJs4k25ziMNvJb43A_with_movement (1.60)
- IrQvqOTCmEWMXn9k2ICtRQ_with_movement (1.58)
- sx6mxKBQpkq520m64BwUdQ_with_movement (1.57)

## 建議 plan 順序調整

原 plan: 組 3 → 組 5 → 組 7 → 組 4 → 組 4.5 (B-15)

**建議調整** (依 top category A≥10 rule):

1. **組 4.5 / B-15 (raw_dt delay-aligned fix)** — 提前到第一優先
   - 理由: 12/20 worst case 都是 non-movement FS onset，symptom 跟 PZ7V 完全同型 (raw_dt 於 echo onset 誤判 DT)
   - 預期影響: top-A 12 case 同步改善，可能收斂大部分 −0.147 退步
2. **組 5 (EPC) + 組 7 (delay tracking)** — 第二優先
   - 理由: 8/20 是 movement case，delay/EPC 相關
   - 組 7 可能部分 overlap B-15 (B-15 需 delay-aligned echo power estimate)
3. **組 3 (saturation)** — 降優先
   - 理由: top-20 零 saturation 案例；B-7/B-9 bisect 顯示影響 < 0.012
4. **組 4 (DTD/RES)** — 降優先
   - 理由: top-20 零 DT 案例；DT_echo 已 −0.081 但 worst case 全在 FS scenario

## 不做

- 不 revert 任何 flag
- 本文件僅為分析 + 建議，待 user confirm 才實作 B-15
