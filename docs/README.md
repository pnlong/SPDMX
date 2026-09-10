# Project page (GitHub Pages)

Everything for the public page lives in this folder. GitHub Pages is configured
to publish **`/docs` on `main`** → **https://pnlong.github.io/SPDMX/**.

| Path | Role |
|------|------|
| `index.html`, `css/`, `js/` | Static site |
| `assets/` | Logo, favicons, chunk-layout figure |
| `data/` | Precomputed JSON for charts |
| `audio/` | Short demo clips + `manifest.json` |
| `export_page_data.py` | Regen JSON / figure / demos from the release |

## Enable Pages

Repo **Settings → Pages → Build and deployment**:

- Source: **Deploy from a branch**
- Branch: `main` / folder `/docs`

## Regenerate data / figures / demos

From the repo root (needs access to the packaged release under `SPDMX_OUTPUT_DIR/SPDMX`):

```bash
.venv/bin/python docs/export_page_data.py
```

Flags:

- `--skip-audio` — JSON + chunk-layout figure only
- `--skip-figure` — skip matplotlib schematic
- `--spdmx-root` / `--dev-root` — override dataset paths

## Logo & favicon

In `assets/`:

- `logo.png` — hero + header mark
- `favicon.ico` / `favicon.svg` / `favicon-96x96.png` — tab icons
- `apple-touch-icon.png`, `web-app-manifest-*.png`, `site.webmanifest`
