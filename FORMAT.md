# Recording format — `marrow.session.v1`

UTF-8 JSONL, one JSON object per line. A session begins with a `session` header and ends
with `end`. Timestamps are integer milliseconds and must be monotonic within a session.
Pointer positions are fractions of the screen (`0..1`), never pixels, so recordings from a
browser, macOS, Windows or a synthetic generator are interchangeable.

```jsonl
{"type":"session","format":"marrow.session.v1","session_id":"s1","user":"opaque","style":"mouse_keyboard","role":"owner_enrol","captured_on":"mac","t":0}
{"type":"move","t":120,"x":0.21,"y":0.44}
{"type":"move","t":190,"x":0.38,"y":0.51}
{"type":"down","t":260,"x":0.40,"y":0.52}
{"type":"up","t":310,"x":0.40,"y":0.52}
{"type":"text","t":700,"source":"typed","op":"insert","chars":3}
{"type":"text","t":940,"source":"typed","op":"replace","chars":2}
{"type":"end","t":1200}
```

| Field | Meaning |
|---|---|
| `type` | `session`, `move`, `down`, `up`, `text`, `end` |
| `t` | integer milliseconds |
| `x`, `y` | screen fraction, `0..1` |
| `source` | `typed`, `on_screen_keyboard`, or `inserted` (dictation/software) |
| `op` | `insert`, `delete`, `replace` |
| `chars` | character count only — **never the characters** |

Rules: no audio, no camera, no device identifiers, no plaintext content. A recording may
come from any capture tool that emits these events.
