async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

/** Friendly labels if JSON is stale. */
function strategyLabel(cat) {
  if (cat.strategy_label) return cat.strategy_label;
  const map = {
    basic: "Basic",
    varied: "Varied",
    ddsp_basic: "MIDI-DDSP (Basic fallback)",
    ddsp_varied: "MIDI-DDSP (Varied fallback)",
  };
  return map[cat.strategy] || cat.strategy || "Basic";
}

const PAPER_CATEGORY_ORDER = [
  "drums",
  "organ",
  "guitar",
  "piano",
  "voice",
  "mallet",
  "polyphonic",
  "strings",
  "brass",
  "wind",
];

const CATEGORY_DISPLAY_LABELS = {
  drums: "Drums",
  organ: "Organ",
  guitar: "Guitar",
  piano: "Piano",
  voice: "Voice",
  mallet: "Mallet",
  polyphonic: "Miscellaneous",
  strings: "Strings",
  brass: "Brass",
  wind: "Wind",
};

function categoryDisplayLabel(cat) {
  return (
    CATEGORY_DISPLAY_LABELS[cat.id] ||
    cat.label ||
    String(cat.id || "").replaceAll("_", " ")
  );
}

function orderCategories(categories) {
  const byId = new Map((categories || []).map((c) => [c.id, c]));
  const ordered = [];
  for (const id of PAPER_CATEGORY_ORDER) {
    if (byId.has(id)) {
      ordered.push(byId.get(id));
      byId.delete(id);
    }
  }
  for (const cat of byId.values()) ordered.push(cat);
  return ordered;
}

function bankListHtml(banks) {
  const rows = (banks || [])
    .map((f) => `<li title="${escapeHtml(f.id || "")}">${escapeHtml(f.file || f.id)}</li>`)
    .join("");
  return `<ul class="soundfont-list">${rows}</ul>`;
}

function bankDropdownHtml({ title, banks, open = false }) {
  const n = (banks || []).length;
  const countLabel = n === 1 ? "1 bank" : `${n} banks`;
  return `
    <details class="soundfont-dropdown"${open ? " open" : ""}>
      <summary>
        <span>${escapeHtml(title)}</span>
        <span class="soundfont-dropdown-count">${countLabel}</span>
      </summary>
      <div class="soundfont-dropdown-body">
        ${bankListHtml(banks)}
      </div>
    </details>
  `;
}

function categoryBlurb(cat) {
  if (cat.uses_midi_ddsp) {
    const fallback =
      cat.fallback === "varied"
        ? "the Varied shortlist below"
        : "the Basic bank below";
    return `MIDI-DDSP is primary when the stem is eligible (monophonic URMP instruments); otherwise we fall back to ${fallback}.`;
  }
  if (cat.strategy === "varied") {
    return `FluidSynth samples one bank from the Varied shortlist below per song.`;
  }
  return `FluidSynth uses the single Basic bank below.`;
}

function categoryCardHtml(cat) {
  const recipe = strategyLabel(cat);
  const name = categoryDisplayLabel(cat);
  const banksTitle = cat.uses_midi_ddsp
    ? `Fallback soundfonts · ${name}`
    : `Soundfonts · ${name}`;
  return `
    <article class="soundfont-category">
      <div class="soundfont-category-head">
        <h3>${escapeHtml(name)}</h3>
        <span class="soundfont-strategy-pill">${escapeHtml(recipe)}</span>
      </div>
      <p class="soundfont-category-blurb">${categoryBlurb(cat)}</p>
      ${bankDropdownHtml({ title: banksTitle, banks: cat.banks })}
    </article>
  `;
}

/** Inventory + per-category recipe with soundfont shortlists. */
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
  if (blurb) {
    blurb.textContent =
      data.caption ||
      "Each listening category uses a locked recipe: Basic, Varied, or MIDI-DDSP with a Basic or Varied soundfont fallback. Open a dropdown for the banks in rotation.";
  }

  const basic = data.basic || {};
  const fonts = data.fonts || [];
  const categories = orderCategories(data.categories || []);
  const n = data.n_varied ?? fonts.length;
  const basicBanks = [{ id: "sgm_v2_01", file: basic.file || "SGM-V2.01.sf2" }];

  const categoryCards = categories.map(categoryCardHtml).join("");

  root.innerHTML = `
    <div class="soundfont-inventory">
      ${bankDropdownHtml({
        title: `Basic · ${basic.file || "SGM-V2.01.sf2"}`,
        banks: basicBanks,
      })}
      ${bankDropdownHtml({
        title: `Varied inventory · ${n} banks`,
        banks: fonts,
      })}
      <p class="soundfont-footnote">
        ${
          data.collection_url
            ? `<a href="${escapeHtml(data.collection_url)}" rel="noopener noreferrer">${escapeHtml(data.collection || "Archive.org collection")}</a>`
            : escapeHtml(data.collection || "Redistributable GM soundfonts")
        }
        · shortlists in <code>winners_locked.yaml</code>
        · recipes in <code>recipe.yaml</code>
      </p>
    </div>
    <section class="soundfont-categories">
      <h3 class="soundfont-section-title">Per-category recipe</h3>
      <p class="soundfont-section-lead">
        For MIDI-DDSP recipes, neural synthesis is attempted first; the listed banks
        are the FluidSynth fallback. Basic categories show the single SGM bank.
      </p>
      <div class="soundfont-category-grid">
        ${categoryCards}
      </div>
    </section>
  `;
}

/** Compact recipe table for the About page (no bank lists). */
export async function renderRecipeTable() {
  const mount = document.getElementById("recipe-table-root");
  if (!mount) return;

  let data;
  try {
    data = await loadJSON("data/soundfonts.json");
  } catch {
    mount.innerHTML =
      "<p class='chart-note'>Recipe table will appear here after export.</p>";
    return;
  }

  const categories = orderCategories(data.categories || []);
  if (!categories.length) {
    mount.innerHTML =
      "<p class='chart-note'>No per-category recipe data in soundfonts.json.</p>";
    return;
  }

  const rows = categories
    .map((cat) => {
      const recipe = strategyLabel(cat);
      const primary = cat.uses_midi_ddsp ? "MIDI-DDSP" : recipe;
      const fallback = cat.uses_midi_ddsp
        ? cat.fallback === "varied"
          ? "Varied shortlist"
          : "Basic (SGM)"
        : "—";
      return `
        <tr>
          <td>${escapeHtml(categoryDisplayLabel(cat))}</td>
          <td>${escapeHtml(recipe)}</td>
          <td>${escapeHtml(primary)}</td>
          <td>${escapeHtml(fallback)}</td>
        </tr>
      `;
    })
    .join("");

  mount.innerHTML = `
    <div class="recipe-table-wrap">
      <table class="recipe-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Recipe</th>
            <th>Primary</th>
            <th>Soundfont fallback</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="chart-note">
      MIDI-DDSP only runs on eligible stems (monophonic URMP instruments); otherwise
      the fallback soundfont is used.
      Bank shortlists are on the <a href="soundfonts.html">Soundfonts</a> page.
    </p>
  `;
}
