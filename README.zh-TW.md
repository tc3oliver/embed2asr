# Embed2ASR — 把音訊 Embedding 變成語音辨識

**凍結的 embedding encoder，加上 9.66M 可訓練的 reader，不微調 encoder 就能轉寫語音。**

[English](README.md) · [繁體中文](README.zh-TW.md) · [快速開始](#快速開始) · [Benchmark](docs/benchmarks.md) · [Hugging Face 模型](https://huggingface.co/tc3oliver/Embed2ASR-v1)

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)
[![CI](https://github.com/tc3oliver/embed2asr/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/embed2asr/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/tc3oliver/embed2asr)](https://github.com/tc3oliver/embed2asr/releases)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Embed2ASR--v1-yellow)](https://huggingface.co/tc3oliver/Embed2ASR-v1)

我們把一個音訊 embedding 模型改用於語音辨識，而且不微調它的 encoder。Embed2ASR 取 Google EmbeddingGemma 2 audio encoder 在 pooling 之前的 `last_hidden_state`（1536 維、每秒 25 個 frame；不是最後的 pooled embedding 向量），交給 9,655,172 參數的 GRU-CTC reader 解碼。

<p align="center"><img src="assets/overview.svg" alt="音訊進入凍結的 EmbeddingGemma 2 audio encoder，pooling 之前的 last_hidden_state（1536 維、每秒 25 frame）交給可訓練的 9,655,172 參數 GRU-CTC reader，輸出逐字稿。原本的 embed_audio 與 pooling 路徑不使用。" width="880"></p>

**為什麼重要。** 預訓練的 embedding 表徵，能不能重用在它原本沒被設計去做的任務上？這個專案針對音訊 embedding 與語音辨識（中文、英文、中英混用）給出一個有實測的答案。

## 效能一覽

<!-- PERF:HERO:BEGIN -->
**batch 1 下，推論速度比 Whisper large-v3 快 4.16 倍、GPU 峰值記憶體低 76.5%，同一張 MI300X、同一批音訊實測。**
<!-- PERF:HERO:END -->

<p align="center"><img src="assets/performance.svg" alt="長條圖：Embed2ASR-v1 與 Whisper large-v3 在單張 MI300X 上的 RTF（batch 1 與 batch 16）與 GPU 峰值記憶體，越低越好。" width="640"></p>

<!-- PERF:TABLE:BEGIN -->
|  | Embed2ASR-v1 | Whisper large-v3 | 差異 |
| --- | --- | --- | --- |
| RTF，batch 1（越低越好） | 0.0138 | 0.0576 | 快 4.16 倍 |
| RTF，batch 16（越低越好） | 0.00345 | 0.0106 | 快 3.06 倍 |
| GPU 峰值記憶體，batch 1 | 0.75 GiB | 3.20 GiB | -76.5% |
| GPU 峰值記憶體，batch 16 | 1.74 GiB | 6.88 GiB | -74.7% |
| 冷啟動（3 次中位數） | 7.9 s | 6.0 s | Whisper 較快 |
<!-- PERF:TABLE:END -->

<!-- PERF:COND:BEGIN -->
測試條件：單張 MI300X，測前閒置；ROCm 7.1、torch 2.10；兩者都用 bf16。200 段音訊（中文 100 + 英文 100），共 1,086 秒。計時包含音檔解碼、特徵、模型與文字輸出，不含模型載入；暖機 1 次後取 3 輪中位數。Whisper 使用 Hugging Face `generate`，greedy（`num_beams=1`）。Embed2ASR 在 batch 16 使用 `cross_file=True`（不同檔案的片段混批；約 4% 的短句可能差 1 個單位；預設關閉）。記憶體是 torch 的 peak allocated，含完整 audio encoder 與 reader，不是整個 process 的用量。冷啟動是從程序啟動到第一個輸出。數字只適用於這兩個 batch size 與這組硬體。轉寫結果在不同 transformers 版本與 GPU 型號之間可能有 ±1 個單位的差異（同一環境內結果一致）；本次 benchmark 環境為 torch 2.10.0+rocm7.1.1、transformers 5.19.0、MI300X。
<!-- PERF:COND:END -->

這是兩個不同系統的比較。差距來自幾個我們沒有做消融的因素（載入的模型大小、CTC 與自回歸解碼的差別、本測試環境），請把它當成這條 pipeline 的實測結果，不要解讀成 embedding 模型本身的特性。多數測試集的準確度低於 Whisper large-v3，見[準確度](#準確度對照-whisper-large-v3)。

完整測試協定與更多數字：[docs/benchmarks.md](docs/benchmarks.md)。

## 架構

Encoder 完全不更新，只訓練 9,655,172 個參數的 reader。推論時載入約 314.5M 參數。

## 準確度：對照 Whisper large-v3

Whisper large-v3 在 Common Voice zh-TW、英文（FLEURS、LibriSpeech）與 NTUML2021 上明顯比較準。Embed2ASR-v1 在 Parliament（兩組 test set）與 ASCEND 的中英混合句上比較準；ASCEND 全部句子合計則是 Whisper 略勝。速度不改變這件事：請依你要轉寫的語音類型選模型。

Embed2ASR-v1 在 Parliament 與 ASCEND 上的優勢，是在與訓練分佈同領域的 test set 上量到的。NTUML2021（ML2021_ASR_ST）是唯一的領域外（out-of-domain）test set，Whisper large-v3 在其上明顯比較準，因此這些優勢不應解讀為跨領域泛化。

Test set，用最終 checkpoint 評估一次。錯誤率，越低越好。

| 資料集 | 指標 | Embed2ASR greedy | + 中文 LM | Whisper large-v3 |
|---|---|---|---|---|
| Common Voice zh-TW 27.0 | CER | <!-- BENCH:cv_zh_tw_cer -->15.6%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_lm -->14.7%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_whisper -->9.6%<!-- /BENCH --> |
| Parliament test（set A） | CER | <!-- BENCH:parl_cer -->19.2%<!-- /BENCH --> | <!-- BENCH:parl_cer_lm -->19.0%<!-- /BENCH --> | <!-- BENCH:parl_cer_whisper -->22.7%<!-- /BENCH --> |
| ASCEND（僅中英混合句） | MER | <!-- BENCH:ascend_mix_mer -->23.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_lm -->22.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_whisper -->27.8%<!-- /BENCH --> |
| NTUML2021（中英混合） | MER | <!-- BENCH:ntuml_mer -->29.5%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_lm -->28.9%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_whisper -->16.0%<!-- /BENCH --> |
| FLEURS en_us | WER | <!-- BENCH:fleurs_en_wer -->24.5%<!-- /BENCH --> | – | <!-- BENCH:fleurs_en_wer_whisper -->4.5%<!-- /BENCH --> |
| LibriSpeech test-clean | WER | <!-- BENCH:libri_clean_wer -->9.0%<!-- /BENCH --> | – | <!-- BENCH:libri_clean_wer_whisper -->4.0%<!-- /BENCH --> |

CER 為字元錯誤率、WER 為詞錯誤率、MER 為中英混合錯誤率（中文以字、英文以詞計），各指標不互相平均。LM 是中文字元 5-gram，所以英文列沒有 LM 欄。Parliament 的參考答案不是逐字稿，CER 因此有一個下限。全部 test set 與硬體：[docs/benchmarks.md](docs/benchmarks.md)。評估上的注意事項見 [docs/limitations.md](docs/limitations.md)。

## 快速開始

推論需要兩樣東西：EmbeddingGemma 2 encoder（依 Google 的條款自行下載），以及 Embed2ASR-v1 的 reader checkpoint。

<!-- QUICKSTART:BEGIN -->
### 安裝

Python 3.10+。先依你的加速器安裝對應的 torch（CUDA、ROCm 或 CPU），再安裝套件：

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1   # 或 CUDA / CPU 版 wheel
pip install "git+https://github.com/tc3oliver/embed2asr.git"
```

EmbeddingGemma 2 encoder 與 Embed2ASR-v1 reader 會在第一次使用時從 Hugging Face 下載。第一次執行會下載 `google/embeddinggemma-2` 的 audio tower；若下載時要求登入，請先執行 `hf auth login`。

### CLI

```bash
embed2asr transcribe audio.wav --model tc3oliver/Embed2ASR-v1
```

選項：`--device cpu|cuda`、`--batch-size N`、`--json`（每個檔案輸出 text、duration、chunks、RTF）。一次可以傳入多個檔案。超過 29.5 秒的音訊會切成連續的片段。

### Python API

```python
from embed2asr import Embed2ASR

asr = Embed2ASR.from_pretrained("tc3oliver/Embed2ASR-v1")  # 有 CUDA/ROCm 就自動使用
result = asr.transcribe("audio.wav")
print(result.text)
```

`transcribe` 也可以傳入路徑 list，或搭配 `sr=` 傳入 float array。回傳的 `Transcript` 有 `text`、`duration`，以及每個片段的 `chunks`。

### 本機 checkpoint

`--model` 與 `from_pretrained` 也接受本機的 release 目錄，內含 `config.json`、`model.safetensors`、`units.json`、`tokenizer_bpe.model`：

```bash
embed2asr transcribe audio.wav --model /path/to/embed2asr-v1
```
<!-- QUICKSTART:END -->

## 試聽

<!-- DEMO:BEGIN -->
三句中英夾雜的語句（華語夾英文單字，3–6 秒），來自開發集。Embed2ASR 為 greedy CTC 輸出；Whisper large-v3 為原始輸出，含它自己的標點與大小寫。前兩句 Embed2ASR 較接近參考答案，最後一句是失敗例。

| reference | Embed2ASR | Whisper large-v3 |
|---|---|---|
| 有我想支持的隊伍的時候我就會very excited | 有我想支持的對伍的時候我就會very excited | 有我想支持的隊伍的時候,我就會非常興奮。 |
| 然後再來就是smart phone它的功能有的時候 | 然後再來就是smart phone他的功能有的時候 | 然後再來就是smartphone它的功能有的時候 |
| 你有抱過koala bear | 你有包過口阿拉伯 | 你有抱過Koala Bear嗎? |

挑選方式：從該集的 121 句候選中手選。這 121 句裡，Embed2ASR 的 MER 低於 Whisper large-v3 的有 53 句、高於的有 59 句、持平 9 句。這個候選池不是 benchmark，分數請看 benchmark 表。
<!-- DEMO:END -->

## 運作方式

1. 編碼：16 kHz 音訊送進 EmbeddingGemma 2 的音訊 encoder（12 層 conformer 加輸出投影，304,824,608 個凍結參數）。
2. 在 pooling 之前取特徵：取 audio tower 的 `last_hidden_state`，每秒 25 幀、每幀 1536 維，位置在 `embed_audio` 與 pooling 之前。
3. 讀出：reader 共 9,655,172 個參數（LayerNorm + Linear 1536→512、三個殘差 Conv1d 區塊、每方向 256 單元的兩層 BiGRU），每一幀預測 4,996 個 CTC 單位之一：blank、4,596 個中文字與其他非 ASCII 字元、399 個英文 BPE 片段。中文字與英文子詞共用同一個輸出層。
4. 解碼：預設 greedy CTC；可選用中文字元 5-gram LM 搭配 prefix beam search。

推論時載入約 314.5M 參數，其中只有 9.66M 的 reader 是訓練出來的。細節見 [docs/architecture.md](docs/architecture.md)、[docs/method.md](docs/method.md)。

## 不只是 ASR

Embedding 模型會把 encoder 狀態 pooling 成一個向量，時間軸因此被丟掉。pooling 之前的狀態仍然帶有時序。Embed2ASR 只驗證一個窄問題：為某個任務（音訊 embedding）訓練出來的表示，能不能用一個小型新 head 讀出它從沒訓練過的任務（語音轉寫）。

如果可以，同樣的做法或許能用在其他有時間軸的任務，例如關鍵字偵測、語言辨識、音訊事件偵測、強制對齊。這裡沒有驗證這些方向，本 repo 只測了語音辨識。

## 限制

- 凍結 encoder 的上限：在訓練資料充分覆蓋的資料集上，看過句子的錯誤率仍接近 dev，表示瓶頸比較像是表示能力。
- 同音字：CTC 每一幀獨立預測，華語錯誤以同音字替換為主（在/再、他/她）。
- 英文：主要錯在功能詞與專有名詞，整體落後 Whisper large-v3。
- 中英混合的證據範圍有限：混合語言的 dev 集很小，且在香港錄製。
- 只支援離線，沒有串流模式。

完整清單與證據：[docs/limitations.md](docs/limitations.md)。

## 重現

訓練與評估：[docs/reproducibility.md](docs/reproducibility.md)。

## Model card 與權重

Embed2ASR-v1 artifacts（reader 權重、tokenizer、unit 清單、可選的 5-gram LM）以 Apache-2.0 釋出。Model card：[docs/model_card.md](docs/model_card.md)。權重：[tc3oliver/Embed2ASR-v1](https://huggingface.co/tc3oliver/Embed2ASR-v1)。EmbeddingGemma 2 encoder 不包含在內。

## 引用

請依 [CITATION.cff](CITATION.cff) 引用本軟體，並依 Google 官方 model card 的說明一併引用 EmbeddingGemma 2。

```bibtex
@software{embed2asr,
  author  = {tc3oliver},
  title   = {Embed2ASR: Can an Embedding Model Become a Speech Recognition Model?},
  version = {0.1.0},
  url     = {https://github.com/tc3oliver/embed2asr},
  year    = {2026}
}
```

## 參與貢獻

最有幫助的是 bug 回報、重現回報與 benchmark 回報。見 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 授權

程式碼與模型權重：Apache-2.0，見 [LICENSE](LICENSE)。EmbeddingGemma 2 encoder 是另外從 Google 下載的元件，依 Google 自己的條款使用。
