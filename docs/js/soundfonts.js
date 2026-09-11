async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

/** Quiet inventory of FluidSynth banks used in the release. */
export async function renderSoundfonts() {
  const root = document.getElementById("soundfonts-root");
  if (!root) return;

  let data;
  try {
    data = await loadJSON("data/soundfonts.json");
  } catch {
    root.innerHTML =
      "<p class='chart-note'>Soundfont inventory will appear here after export.</p>";
    return;
  }

  const blurb = document.getElementById("soundfonts-blurb");
  if (blurb && data.caption) blurb.textContent = data.caption;

  const basic = data.basic || {};
  const fonts = data.fonts || [];
  const n = data.n_varied ?? fonts.length;

  const items = fonts
    .map((f) => `<li title="${f.id || ""}">${f.file || f.id}</li>`)
    .join("");

  root.innerHTML = `
    <p class="soundfont-basic">
      <strong>Basic</strong>
      <code>${basic.file || "SGM-V2.01.sf2"}</code>
      ${basic.note ? ` · ${basic.note}` : ""}
    </p>
    <details class="soundfont-panel">
      <summary>Varied inventory · ${n} banks</summary>
      <div class="soundfont-panel-body">
        <ul class="soundfont-grid">${items}</ul>
        <p class="soundfont-footnote">
          ${
            data.collection_url
              ? `<a href="${data.collection_url}" rel="noopener noreferrer">${data.collection || "Archive.org collection"}</a>`
              : data.collection || "Redistributable GM soundfonts"
          }
          · locked shortlists in <code>winners_locked.yaml</code>
        </p>
      </div>
    </details>
  `;
}
