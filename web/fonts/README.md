# Fonts

Self-hosted web fonts, so the app looks the same on every system and makes no requests to
font services. Each font is split by Unicode range: browsers download only the pieces that the
page actually uses.

| Directory | Font | Used for | Package | License |
| --- | --- | --- | --- | --- |
| `noto-sans-kr/` | Noto Sans KR (variable weight) | Korean and Latin text | `@fontsource-variable/noto-sans-kr` 5.3.1 | SIL OFL 1.1 |
| `noto-sans-jp/` | Noto Sans JP (variable weight) | Japanese text | `@fontsource-variable/noto-sans-jp` 5.3.1 | SIL OFL 1.1 |
| `jetbrains-mono/` | JetBrains Mono (variable weight) | Numbers, status lines, the mock terminal | `@fontsource-variable/jetbrains-mono` 5.3.0 | SIL OFL 1.1 |

The files are the Google Fonts builds as packaged by [Fontsource](https://fontsource.org/):
`font.css` is the package's `wght.css`, and `files/` holds the upright `woff2` files it
references. `OFL.txt` is the license file from the
[google/fonts](https://github.com/google/fonts) repository, with the fonts' copyright and
reserved names. The fonts are unmodified.

To update, unpack the newer packages (`npm pack @fontsource-variable/<name>`) and copy the
same files again.
