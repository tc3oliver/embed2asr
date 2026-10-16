# Demo

Three code-switching utterances (Mandarin with English words, 3-6 s) from a development set. Embed2ASR output is greedy CTC
from this package; Whisper large-v3 is the unmodified output from the same utterances. Text is shown as produced, including
Whisper's punctuation and casing. The first two are cases where Embed2ASR is closer to the reference, the last is a failure.

| reference | Embed2ASR | Whisper large-v3 |
|---|---|---|
| 有我想支持的隊伍的時候我就會very excited | 有我想支持的對伍的時候我就會very excited | 有我想支持的隊伍的時候,我就會非常興奮。 |
| 然後再來就是smart phone它的功能有的時候 | 然後再來就是smart phone他的功能有的時候 | 然後再來就是smartphone它的功能有的時候 |
| 你有抱過koala bear | 你有包過口阿拉伯 | 你有抱過Koala Bear嗎? |

Selection: examples were picked by hand from 121 candidate utterances of that set. On those 121, Embed2ASR has a lower
MER than Whisper large-v3 on 53 and a higher one on 59 (9 ties); the pool is not a benchmark, see the evaluation table for scores.
