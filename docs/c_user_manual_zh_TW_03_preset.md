> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_02_config.md](c_user_manual_zh_TW_02_config.md) · 下一部分：[c_user_manual_zh_TW_04_seam_troubleshooting.md](c_user_manual_zh_TW_04_seam_troubleshooting.md) <!-- split-nav -->
## 7. Preset 選擇

```c
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
```

mild / balanced 兩個 preset **只差一個欄位**：`min_gain_floor_far_active_db`。
aggressive 則**再加一個欄位**：`min_gain_floor_dt_db`（double-talk 時改用的
地板）——否則 double-talk 偵測觸發時（遠端單講時也會誤觸發）三個
preset 會被硬蓋回同一個 `-20.0`，聽不出 preset 之間的強度差。這三個 preset
其餘欄位完全相同。第四個 preset `no_floor`
（`AEC_PRESET_NO_FLOOR`，enum 值 `3`）不是再調深同一個 dB 值，而是把地板
機制整個關掉：第三個欄位 `min_gain_split_floor_enabled` 設為 `0`，這時前兩個
dB 欄位變成**惰性值**（維持 balanced 的 bytes，但 `get_min_gain` 的三個
`if (c->split_floor_enabled)` 分支都不會再讀它們）。所以完整的 preset
軸是「回音壓多乾淨」對「近端保留多完整」的單一取捨軸，`mild`／`balanced`／
`aggressive` 用兩個 dB 欄位表達，`no_floor` 則是這條軸上最深的一個操作點
（地板關閉，gain 純粹由 audibility/R² 決定）。

| Preset | `min_gain_floor_far_active_db` | `min_gain_floor_dt_db` | `min_gain_split_floor_enabled` | 適用情境 |
|---|---:|---:|---:|---|
| `AEC_PRESET_MILD` | `-20.0` | `-20.0` | `1` | 近端音質優先。喇叭音量小、耦合弱的場景；可接受較多回音殘留。 |
| `AEC_PRESET_BALANCED` | `-28.0` | `-20.0` | `1` | **預設，一般通話建議值。** |
| `AEC_PRESET_AGGRESSIVE` | `-38.0` | `-30.0` | `1` | 回音抑制優先。大音量、喇叭與麥克風靠很近、耦合強的場景；double-talk 時近端較容易被壓抑，這是刻意的。 |
| `AEC_PRESET_NO_FLOOR` | `-28.0`（惰性） | `-20.0`（惰性） | `0` | 這條軸上最深的操作點：三種地板（far-active、far-silent/純近端、DT-lifted）全部關閉，近端損失最大，換回最多回音抑制。 |

先用 preset 調整，不要一開始就自己填 `min_gain_floor_far_active_db` /
`min_gain_floor_dt_db` / `min_gain_split_floor_enabled`。需要中間值時，才
直接覆寫這些欄位。傳入超出 enum 範圍的 preset 值會退回 balanced 的設定
（三個欄位都退回，含 `min_gain_split_floor_enabled = 1`）。

### 7.1 執行期切換 preset（`aec_set_preset`）

```c
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);
```

不需要重建實例。換 preset 是**重新指定目標值**，不是重建：濾波器、延遲狀態、
每一條平滑歷史、far-active latch 與 DominantNearend 計數器全部原封不動繼續
跑。但三個欄位走不同的路：`min_gain_floor_far_active_db` 是抑制器每個 hop
只讀一次的純量下限，`ramp_ms` 讓它用 dB 線性走到新值；`min_gain_floor_dt_db`
是 double-talk 時的硬性覆寫，不是逐 hop 收斂的量，沒有 ramp 可言，不論
`ramp_ms` 是多少都**立即套用**；`min_gain_split_floor_enabled` 是布林值，同樣
**立即套用**（布林沒有連續路徑可以 ramp，與 DT 地板是同一種紀律，只是多套
在第三個欄位上）——C 跟 Python 兩邊語意相同（Python 的 `AEC.set_preset()`
同樣開放執行期切換 `no_floor`，見本節最後一段）。

切到 `no_floor`（`enabled: True -> False`）時，閂住的 gate 立即關閉；原本
可能還在走的 far-active ramp 不受影響地繼續跑向它的舊目標，但因為三個
`if (c->split_floor_enabled)` 分支都不再讀它，這個繼續跑的 ramp 是**無效但
無害**的。從 `no_floor` 切回有地板的 preset（`enabled: False -> True`）時，
gate 立即打開，**同時**把 far-active／dt 兩個 dB 欄位也透過既有的 setter
寫成目標 preset 的值——不會出現「gate 開了但 dB 還停在 no_floor 的惰性值」
這種半套用狀態。唯一不處理的細節：`far_active_latched`（far-active latch）
在 `no_floor` 期間完全不會被改動（閂不住，因為沒人讀它），切回來之後也
**不會**被強制重置或強制觸發，就從原本的狀態繼續——如果一段 `no_floor`
期間遠端一直很大聲，切回來後的前幾個 hop 仍可能先走 far-silent／純近端
的地板分支，等 latch 再次被觸發才換成 far-active 分支。

```c
#include "aec.h"

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
Aec* a = aec_init(pool, aec_get_mem_size(&cfg), &cfg);

for (;;) {
    /* 使用者把強度旋鈕轉到 aggressive：在兩個 hop 之間呼叫，
     * 100 ms 內以 dB 線性走到新的地板 */
    if (ui_strength_changed())
        (void)aec_set_preset(a, AEC_PRESET_AGGRESSIVE, 100.0f);

    aec_process(a, mic, ref, out);   /* 各恰好 hop 個 float32 */
}
```

`ramp_ms` 語意：

| `ramp_ms` | 行為 |
|---|---|
| `0.0f` | 下一個 hop 就套用，**不是錯誤**。落點與「用該 preset 從頭建一個新實例」持有的 float 完全相同 |
| `> 0.0f` | 以 dB 為單位線性走過去，每次 hop 走一步，最後**精確落在**目標值（不是漸近逼近）。上限 60 秒，超過即拒絕 |

有 ramp 的理由：mild ↔ aggressive 是 18 dB 的落差，而這個地板是硬性 clamp，
下游沒有任何東西會替它平滑。100 ms 是合理的起點，但這是呼叫端的選擇，
函式庫沒有寫死任何政策。

**回傳與拒絕**：成功回 `0`。以下情況回 `-1` 且**什麼都不寫**——`a == NULL`、
`preset` 超出 enum 範圍、`ramp_ms` 非有限值或超出 `[0, 60000]`。注意這裡與
`aec_config_from_preset()` 的差異：那是 config factory，遇到不認得的值**退回
balanced**；這是 setter，有呼叫端可以回報，所以**直接拒絕**。

**呼叫紀律**：在兩個 hop 之間呼叫，與 `aec_process()`（或你實際使用的處理進入點）
序列化。**非 thread-safe**。ramp 進行中再呼叫一次，會從**當前的 live 值**重新起走，
既不會彈回舊目標、也不會沿用舊的步進比例。`aec_reset()` 保留目標值、丟棄 ramp 進度
（目標存在 config、進度存在 state，這是刻意分開的）。

**底層 primitive**。真正做事的是抑制器自己的入口。四麥克風產品共用的那個 post 級
抑制器是一個 `AecRes`（沒有任何 `Aec` 擁有它；Audio_ALG 的 4ch 單元也不含它，由應用在
單元之後自己 `aec_res_init()` 並持有）；對它的入口是
`aec_res_set_preset()`（[§8.5](c_user_manual_zh_TW_04_seam_troubleshooting.md#85-獨立的-res-級aec_resh)），語意與 `aec_set_preset()` 相同。`AecRes` 內部包的
是 `SuppressionGain`，其底層函式為（內部 header，整合端不需要直接呼叫）：

```c
#include "suppression_gain.h"   /* 內部 header；一般整合改用 aec_res_set_preset() */

int suppression_gain_set_split_floor_far_active_db(SuppressionGain* sg,
                                                   float db, float ramp_ms);
int suppression_gain_set_split_floor_dt_db(SuppressionGain* sg, float db);
int suppression_gain_set_split_floor_enabled(SuppressionGain* sg, int enabled);
```

`db` 分別對應 `AecConfig.min_gain_floor_far_active_db` / `min_gain_floor_dt_db`
那兩個 dB-power 量，並且用**驗證器自己的**範圍 `[-300, 50]` 檢查——執行期呼叫
不可能裝進一個 init 當初會拒絕的值。遠端地板的 `ramp_ms` 語意與上表相同；DT
地板**沒有** `ramp_ms` 參數，因為它是逐 hop 的硬性覆寫而不是逐 hop 收斂的量，
呼叫即立即套用。`enabled` 對應 `min_gain_split_floor_enabled`，同樣沒有
`ramp_ms`（布林沒有連續路徑），任何非 0 值都會被 coerce 成 `1`；只有
`sg == NULL` 才回 `-1`，其餘輸入一律接受（不像前兩個函式有浮點範圍要拒絕）。
三者都不動任何平滑歷史、latch、DominantNearend 計數器或 `initial_state`。
前兩者回 `0` 或 `-1`（NULL 或引數超出範圍），`-1` 時什麼都不寫：所有值都
先算完、驗完，才寫第一個 store。

preset 查表函式有三個：`aec_preset_floor_db()`、`aec_preset_dt_floor_db()` 與
`aec_preset_split_floor_enabled(AecPreset, int*)`，契約相同（in-enum 寫 `*enabled_out` 並回 `0`；out-of-enum 什麼都不寫並回 `-1`）——
四麥克風產品的共用 post 級抑制器可以只讀這個 gate，不必連帶查兩個 dB 值。

**四麥克風產品請勿對 lane 呼叫**。四條 lane 都以 `spatial_linear_context` 建立，
根本不會走到 `suppression_gain_get_gain()`，所以對 lane 呼叫 `aec_set_preset()`
**依建構方式即為無效操作**——塑形輸出的 gain 來自該產品自己那一個共用的 post 級
抑制器。請對那個 `AecRes` 呼叫 `aec_res_set_preset(res, preset, ramp_ms)`（[§8.5](c_user_manual_zh_TW_04_seam_troubleshooting.md#85-獨立的-res-級aec_resh)）——
`res` 是應用在單元之後自己建立的那一個——它針對的就是那個共用的 post 級抑制器。

**A/B 量測時該預期什麼**（先讀這段再下結論）：far-active 地板只在
**far-active 且非 double-talk** 的 hop 上生效。double-talk 期間套用的是 DT 地板
（mild/balanced 共用 `-20.0`，aggressive 是 `-30.0`）；
far-active latch 尚未觸發之前，套用的是 far-silent 地板。同一個 gain 還決定注入的 comfort noise 量
（振幅正比於 `sqrt(1 − G_res²)`，所以地板壓得越深、CNG 反而越多）。三件事合起來
的結果是：**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊
或 degradation 對齊的條件下比較，並實際試聽。

Python 對應：`AEC.set_preset(preset, ramp_ms=0.0)`（同樣的語意；不合法的 preset 或
`ramp_ms` 丟 `ValueError`，實例不變）。C/Python 兩邊對稱：
`aec_set_preset()` / `aec_res_set_preset()` 與 Python 的 `AEC.set_preset()`
都允許執行期切到/切出 `NO_FLOOR`（或 `no_floor`），沒有任何拒絕，兩邊都是
gate 立即階躍、dB 欄位照既有 setter 走（見上方關於 ramp 與立即套用的段落）。
建構時的 `no_floor`（`aec_config_from_preset()` / `AecConfig.from_preset()`）
與執行期切換在兩邊語意一致。回歸測試：`make test-runtime-preset`（含
NO_FLOOR 切換與 round-trip 的覆蓋）與 `python/tests/test_runtime_preset.py`
的 `test_set_preset_allows_no_floor` 等。

---

> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_02_config.md](c_user_manual_zh_TW_02_config.md) · 下一部分：[c_user_manual_zh_TW_04_seam_troubleshooting.md](c_user_manual_zh_TW_04_seam_troubleshooting.md) <!-- split-nav -->
