[C 使用手冊](../c_user_manual_zh_TW.md)

## 7. Level 選擇

C 稱這條強度軸為 **level**（`AecLevel`、`AEC_LEVEL_*`、`--level`），Python 對應物稱為 **preset**（`AecPreset`、`AecConfig.from_preset`、`--preset`）：**C level == Python preset**，名稱（`mild`／`balanced`／`aggressive`／`no_floor`）與數字（0..3，預設 `1` = balanced）兩邊相同。Python 測試以文字讀取 `aec.h` 的 `AEC_LEVEL_*` 四行；CLI 的 `--preset` 仍可作為 `--level` 的別名。

```c
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);   /* 存進 cfg.level */
```

`AecConfig.level` 是唯一的強度設定，只收數字（0..3，範圍外被拒絕）。函式庫由它查表帶出三個內部值（庫內部 `AecTuning`，
`aec_level_floors()` 可讀），呼叫端不能單獨設定：
mild / balanced 兩個 level **只差一個值**：`min_gain_floor_far_active_db`。
aggressive 則**再加一個值**：`min_gain_floor_dt_db`（double-talk 時改用的
地板）——否則 double-talk 偵測觸發時（遠端單講時也會誤觸發）三個
level 會被硬蓋回同一個 `-20.0`，聽不出 level 之間的強度差。這三個 level
其餘值完全相同。第四個 level `no_floor`
（`AEC_LEVEL_NO_FLOOR`，enum 值 `3`）不是再調深同一個 dB 值，而是把地板
機制整個關掉：第三個值 `min_gain_split_floor_enabled` 設為 `0`，這時前兩個
dB 值變成**惰性值**（維持 balanced 的 bytes，但 `get_min_gain` 的三個
`if (c->split_floor_enabled)` 分支都不會再讀它們）。所以完整的 level
軸是「回音壓多乾淨」對「近端保留多完整」的單一取捨軸，`mild`／`balanced`／
`aggressive` 用兩個 dB 值表達，`no_floor` 則是這條軸上最深的一個操作點
（地板關閉，gain 純粹由 audibility/R² 決定）。

| Level | `min_gain_floor_far_active_db`（內部） | `min_gain_floor_dt_db`（內部） | `min_gain_split_floor_enabled`（內部） | 適用情境 |
|---|---:|---:|---:|---|
| `AEC_LEVEL_MILD` | `-20.0` | `-20.0` | `1` | 近端音質優先。喇叭音量小、耦合弱的場景；可接受較多回音殘留。 |
| `AEC_LEVEL_BALANCED` | `-28.0` | `-20.0` | `1` | **預設，一般通話建議值。** |
| `AEC_LEVEL_AGGRESSIVE` | `-38.0` | `-30.0` | `1` | 回音抑制優先。大音量、喇叭與麥克風靠很近、耦合強的場景；double-talk 時近端較容易被壓抑，這是刻意的。 |
| `AEC_LEVEL_NO_FLOOR` | `-28.0`（惰性） | `-20.0`（惰性） | `0` | 這條軸上最深的操作點：三種地板（far-active、far-silent/純近端、DT-lifted）全部關閉，近端損失最大，換回最多回音抑制。 |

強度只能用 level 調整：三個內部值不是 `AecConfig` 欄位，沒有「中間值」；需要新的操作點時是新增一個 level，而不是讓呼叫端填數字。
`aec_config_from_level()` 收到超出 enum 範圍的 level 值時存 `AEC_LEVEL_BALANCED`（三個值都退回 balanced，含 split floor 開）；
直接把 `AecConfig.level` 填成範圍外的數字，建立時被拒絕。

### 7.1 執行期切換 level（`aec_set_level`）

```c
int aec_set_level(Aec* a, AecLevel level);
```

不需要重建實例。換 level 是**重新指定目標值**，不是重建：濾波器、延遲狀態、
每一條平滑歷史、far-active latch 與 DominantNearend 計數器全部原封不動繼續
跑。但三個值走不同的路：`min_gain_floor_far_active_db` 是抑制器每個 hop
只讀一次的純量下限，函式庫固定用 100 ms 讓它以 dB 線性走到新值；`min_gain_floor_dt_db`
是 double-talk 時的硬性覆寫，不是逐 hop 收斂的量，沒有 ramp 可言，
**立即套用**；`min_gain_split_floor_enabled` 是布林值，同樣
**立即套用**（布林沒有連續路徑可以 ramp，與 DT 地板是同一種紀律，只是多套
在第三個欄位上）——C 跟 Python 兩邊語意相同（Python 的 `AEC.set_preset()`
同樣開放執行期切換 `no_floor`，見本節最後一段）。切換成功後，實例記錄新的 level（`aec_apply_level()`：單聲道與陣列實例共用同一個寫入點）；失敗（`-1`）時什麼都不寫。

切到 `no_floor`（`enabled: True -> False`）時，閂住的 gate 立即關閉；原本
可能還在走的 far-active ramp 不受影響地繼續跑向它的舊目標，但因為三個
`if (c->split_floor_enabled)` 分支都不再讀它，這個繼續跑的 ramp 是**無效但
無害**的。從 `no_floor` 切回有地板的 level（`enabled: False -> True`）時，
gate 立即打開，**同時**把 far-active／dt 兩個 dB 欄位也透過既有的 setter
寫成目標 level 的值——不會出現「gate 開了但 dB 還停在 no_floor 的惰性值」
這種半套用狀態。唯一不處理的細節：`far_active_latched`（far-active latch）
在 `no_floor` 期間完全不會被改動（閂不住，因為沒人讀它），切回來之後也
**不會**被強制重置或強制觸發，就從原本的狀態繼續——如果一段 `no_floor`
期間遠端一直很大聲，切回來後的前幾個 hop 仍可能先走 far-silent／純近端
的地板分支，等 latch 再次被觸發才換成 far-active 分支。

```c
#include "aec.h"

AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
Aec* a = aec_init(pool, aec_get_mem_size(&cfg), &cfg);

for (;;) {
    /* 使用者把強度旋鈕轉到 aggressive：在兩個 hop 之間呼叫，
     * 地板在 100 ms 內以 dB 線性走到新值 */
    if (ui_strength_changed())
        (void)aec_set_level(a, AEC_LEVEL_AGGRESSIVE);

    aec_process(a, mic, ref, out);   /* 各恰好 hop 個 float32 */
}
```

**ramp**：far-active 地板每個 hop 走一步，在 100 ms 內以 dB 線性走過去，最後**精確落在**目標值
（不是漸近逼近）；落點與「用該 level 從頭建一個新實例」持有的 float 完全相同。函式庫把 100 ms 寫死為內部常數，
呼叫端不能調。有 ramp 的理由：mild ↔ aggressive 是 18 dB 的落差，而這個地板是硬性 clamp，
下游沒有任何東西會替它平滑。需要「下一個 hop 就生效」的測試，改走抑制器自己的
`suppression_gain_set_split_floor_far_active_db(sg, db, 0.0f)`（見下方底層 primitive）。

**回傳與拒絕**：成功回 `0`。以下情況回 `-1` 且**什麼都不寫**——`a == NULL`、
`level` 超出 enum 範圍。注意這裡與
`aec_config_from_level()` 的差異：那是 config factory，遇到不認得的值**退回
balanced**；這是 setter，有呼叫端可以回報，所以**直接拒絕**。

**呼叫紀律**：在兩個 hop 之間呼叫，與 `aec_process()`（或你實際使用的處理進入點）
序列化。**非 thread-safe**。ramp 進行中再呼叫一次，會從**當前的 live 值**重新起走，
既不會彈回舊目標、也不會沿用舊的步進比例。`aec_reset()` 保留目標值、丟棄 ramp 進度
（目標存在 config、進度存在 state，這是刻意分開的）。

**底層 primitive**。真正做事的是抑制器自己的入口。四麥克風產品共用的那個 post 級
抑制器在陣列實例的記憶池內（`AecConfig.n_mics = 4` 且 `residual = FULL`，[§8.8](integration.md#88-麥克風陣列實例n_mics--4)）；
對它的入口就是 `aec_set_level()`，語意不變。它內部包的
是 `SuppressionGain`，其底層函式為（內部 header，整合端不需要直接呼叫）：

```c
#include "suppression_gain.h"   /* 內部 header（c_impl/src/，不在 include/）；一般整合改用 aec_set_level() */

int suppression_gain_set_split_floor_far_active_db(SuppressionGain* sg,
                                                   float db, float ramp_ms);
int suppression_gain_set_split_floor_dt_db(SuppressionGain* sg, float db);
int suppression_gain_set_split_floor_enabled(SuppressionGain* sg, int enabled);
```

`db` 分別對應內部的 `min_gain_floor_far_active_db` / `min_gain_floor_dt_db`
那兩個 dB-power 量，並且用**驗證器自己的**範圍 `[-300, 50]` 檢查——執行期呼叫
不可能裝進一個 init 當初會拒絕的值。遠端地板的 `ramp_ms` 在 `[0, 60000]` 內，`0` 是下一個 hop 就套用，大於 0 以 dB 線性走到目標（`aec_set_level()` 固定傳 100）；DT
地板**沒有** `ramp_ms` 參數，因為它是逐 hop 的硬性覆寫而不是逐 hop 收斂的量，
呼叫即立即套用。`enabled` 對應 `min_gain_split_floor_enabled`，同樣沒有
`ramp_ms`（布林沒有連續路徑），任何非 0 值都會被 coerce 成 `1`；只有
`sg == NULL` 才回 `-1`，其餘輸入一律接受（不像前兩個函式有浮點範圍要拒絕）。
三者都不動任何平滑歷史、latch、DominantNearend 計數器或 `initial_state`。
前兩者回 `0` 或 `-1`（NULL 或引數超出範圍），`-1` 時什麼都不寫：所有值都
先算完、驗完，才寫第一個 store。

level 查表函式是 `aec_level_floors(AecLevel, AecLevelFloors*)`（函式庫內部，宣告在 `aec_internal.h`）：in-enum 的 level 寫入 `far_active_db`、`dt_db`、`split_floor_enabled` 三個欄位並回 `0`；out-of-enum 或 `out == NULL` 什麼都不寫並回 `-1`
（`AEC_LEVEL_NO_FLOOR` 的 `far_active_db = -28.0`、`dt_db = -20.0` 只是惰性佔位值，不是該 level 的地板）。

**陣列實例的 lane**。四條 lane 的 `AecConfig.residual` 都低於 `AEC_RESIDUAL_FULL`（`STATE`／`ESTIMATE`），
根本不會走到 `suppression_gain_get_gain()`，所以 lane 自己的強度**依建構方式即為無效操作**——塑形輸出的 gain 來自
融合後那一個共用的 post 級抑制器。對陣列實例呼叫 `aec_set_level(a, level)` 就是重設那個抑制器的強度
（lane 不受影響）。

**A/B 量測時該預期什麼**（先讀這段再下結論）：far-active 地板只在
**far-active 且非 double-talk** 的 hop 上生效。double-talk 期間套用的是 DT 地板
（mild/balanced 共用 `-20.0`，aggressive 是 `-30.0`）；
far-active latch 尚未觸發之前，套用的是 far-silent 地板。同一個 gain 還決定注入的 comfort noise 量
（振幅正比於 `sqrt(1 − G_res²)`，所以地板壓得越深、CNG 反而越多）。三件事合起來
的結果是：**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊
或 degradation 對齊的條件下比較，並實際試聽。

Python 對應：`AEC.set_preset(preset, ramp_ms=0.0)`（Python 孿生保留 `ramp_ms` 參數，不合法的 preset 或
`ramp_ms` 丟 `ValueError`，實例不變；C 端 ramp 固定 100 ms，等同 Python 傳 `ramp_ms=100.0`）。C/Python 兩邊對稱：
`aec_set_level()` 與 Python 的 `AEC.set_preset()`
都允許執行期切到/切出 `NO_FLOOR`（或 `no_floor`），沒有任何拒絕，兩邊都是
gate 立即階躍、dB 欄位照既有 setter 走（見上方關於 ramp 與立即套用的段落）。
建構時的 `no_floor`（`aec_config_from_level()` / `AecConfig.from_preset()`）
與執行期切換在兩邊語意一致。回歸測試：`make test-runtime-preset`（含
NO_FLOOR 切換與 round-trip 的覆蓋）與 `python/tests/test_runtime_preset.py`
的 `test_set_preset_allows_no_floor` 等。

---
