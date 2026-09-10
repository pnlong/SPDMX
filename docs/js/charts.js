const INK = "#1a1510";
const MUTED = "#6a5a48";

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: {
      labels: {
        color: INK,
        boxWidth: 12,
        font: { family: "'IBM Plex Sans', sans-serif", size: 11 },
      },
    },
    tooltip: {
      backgroundColor: "rgba(26, 21, 16, 0.92)",
      titleFont: { family: "'IBM Plex Sans', sans-serif" },
      bodyFont: { family: "'IBM Plex Sans', sans-serif" },
    },
  },
  scales: {
    x: {
      ticks: { color: MUTED, font: { size: 10 } },
      grid: { color: "rgba(26, 21, 16, 0.08)" },
      border: { color: "rgba(26, 21, 16, 0.18)" },
    },
    y: {
      ticks: { color: MUTED, font: { size: 10 } },
      grid: { color: "rgba(26, 21, 16, 0.08)" },
      border: { color: "rgba(26, 21, 16, 0.18)" },
    },
  },
};

const PALETTE = {
  marigold: "#f0b068",
  marigoldSoft: "rgba(240, 176, 104, 0.82)",
  amber: "#e8922e",
  amberDeep: "#c46a12",
  ink: INK,
  bronze: "#8a6238",
  sand: "#d4a574",
};

function fmtCount(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString("en-US");
}

/** Compact tick labels: 219977 → 220k, 432620 → 433k */
function fmtCompact(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) < 1000) return String(Math.round(v));
  if (Math.abs(v) < 1_000_000) {
    const k = v / 1000;
    const rounded = k >= 100 ? Math.round(k) : Math.round(k * 10) / 10;
    return `${rounded.toLocaleString("en-US")}k`;
  }
  const m = v / 1_000_000;
  const rounded = Math.round(m * 100) / 100;
  return `${rounded.toLocaleString("en-US")}M`;
}

function prettyLabel(raw) {
  const key = String(raw || "");
  const special = {
    slakh: "Varied soundfonts",
    "Slakh-style (FluidSynth)": "Varied soundfonts",
    "Varied soundfonts": "Varied soundfonts",
    "Varied (FluidSynth)": "Varied soundfonts",
    "Basic (FluidSynth)": "Basic (FluidSynth)",
    "MIDI-DDSP": "MIDI-DDSP",
    basic: "Basic (FluidSynth)",
    midi_ddsp: "MIDI-DDSP",
    "midi-ddsp": "MIDI-DDSP",
    fluidsynth: "FluidSynth",
    ddsp_piano: "DDSP-Piano",
    chromatic_percussion: "Chromatic percussion",
    synth_pad: "Synth pad",
    synth_lead: "Synth lead",
    synth_effects: "Synth effects",
    sound_effects: "Sound effects",
  };
  if (special[key]) return special[key];
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

function barChart(canvasId, labels, data, opts = {}) {
  const el = document.getElementById(canvasId);
  if (!el || typeof Chart === "undefined") return;
  const horizontal = Boolean(opts.horizontal);
  const colors = opts.colors || opts.color || PALETTE.marigoldSoft;
  const niceTicks = Boolean(opts.compactTicks);
  const valueAxis = horizontal ? "x" : "y";
  const categoryAxis = horizontal ? "y" : "x";
  const valueName = opts.valueLabel || opts.label || "Count";

  const scales = {
    ...CHART_DEFAULTS.scales,
    [valueAxis]: {
      ...CHART_DEFAULTS.scales[valueAxis],
      ticks: {
        ...CHART_DEFAULTS.scales[valueAxis].ticks,
        callback: niceTicks ? (v) => fmtCompact(v) : (v) => fmtCount(v),
      },
    },
    [categoryAxis]: {
      ...CHART_DEFAULTS.scales[categoryAxis],
    },
  };

  return new Chart(el, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: opts.label || valueName,
          data,
          backgroundColor: colors,
          borderWidth: 0,
          borderRadius: 4,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: horizontal ? "y" : "x",
      scales,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: Boolean(opts.showLegend) },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            label(ctx) {
              const raw = ctx.parsed[valueAxis] ?? ctx.raw;
              return ` ${valueName}: ${fmtCount(raw)}`;
            },
          },
        },
      },
    },
  });
}

function doughnutChart(canvasId, labels, data) {
  const el = document.getElementById(canvasId);
  if (!el || typeof Chart === "undefined") return;
  const colors = [
    PALETTE.marigold,
    PALETTE.amber,
    PALETTE.bronze,
    PALETTE.sand,
    PALETTE.amberDeep,
  ];
  const total = data.reduce((a, b) => a + Number(b || 0), 0) || 1;
  const pretty = labels.map(prettyLabel);

  return new Chart(el, {
    type: "doughnut",
    data: {
      labels: pretty,
      datasets: [
        {
          data,
          backgroundColor: labels.map((_, i) => colors[i % colors.length]),
          borderWidth: 0,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {},
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: {
          position: "bottom",
          labels: {
            ...CHART_DEFAULTS.plugins.legend.labels,
            generateLabels(chart) {
              const ds = chart.data.datasets[0];
              return chart.data.labels.map((label, i) => {
                const value = Number(ds.data[i] || 0);
                const pct = Math.round((value / total) * 1000) / 10;
                return {
                  text: `${label}  ${fmtCompact(value)} (${pct}%)`,
                  fillStyle: ds.backgroundColor[i],
                  strokeStyle: ds.backgroundColor[i],
                  hidden: false,
                  index: i,
                };
              });
            },
          },
        },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            label(ctx) {
              const value = Number(ctx.raw || 0);
              const pct = Math.round((value / total) * 1000) / 10;
              return ` ${ctx.label}: ${fmtCount(value)} (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

function audioPeers(comparison) {
  // Symbolic corpora (e.g. PDMX) are not audio peers.
  return Object.entries(comparison || {}).filter(
    ([name, row]) =>
      String(name).toUpperCase() !== "PDMX" &&
      String(row?.type || "").toLowerCase() !== "symbolic"
  );
}

export async function renderCharts() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.backgroundColor = "transparent";
  Chart.defaults.color = INK;

  const [comparison, tracks, gm, programs, backends, chunks, duration, summary] =
    await Promise.all([
      loadJSON("data/comparison.json").catch(() => ({})),
      loadJSON("data/tracks_per_song.json").catch(() => null),
      loadJSON("data/gm_classes.json").catch(() => null),
      loadJSON("data/programs_top.json").catch(() => null),
      loadJSON("data/backends.json").catch(() => null),
      loadJSON("data/chunks.json").catch(() => null),
      loadJSON("data/duration.json").catch(() => null),
      loadJSON("data/summary.json").catch(() => null),
    ]);

  if (summary) {
    const fmt = (n) =>
      typeof n === "number"
        ? n >= 1000
          ? n.toLocaleString("en-US")
          : String(n)
        : "—";
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
    set("stat-songs", fmt(summary.release_songs));
    set("stat-stems", fmt(summary.release_stems));
    set(
      "stat-hours",
      summary.release_hours_approx
        ? `~${Number(summary.release_hours_approx).toLocaleString("en-US")}`
        : "—"
    );
    set("stat-chunks", fmt(summary.n_chunks));
  }

  const peers = audioPeers(comparison);
  if (peers.length) {
    const names = peers.map(([k]) => k);
    barChart(
      "chart-comparison",
      names,
      peers.map(([, row]) => row.hours ?? 0),
      {
        label: "Hours",
        valueLabel: "Hours",
        showLegend: true,
        compactTicks: true,
        colors: names.map((n) =>
          String(n).toLowerCase() === "spdmx" ? PALETTE.amber : PALETTE.marigoldSoft
        ),
      }
    );
  }

  if (tracks) {
    barChart("chart-tracks", tracks.labels, tracks.counts, {
      color: PALETTE.sand,
      compactTicks: true,
      valueLabel: "Songs",
    });
    const note = document.getElementById("tracks-note");
    if (note) {
      const nSingle = tracks.n_single ?? 0;
      const frac = tracks.frac_single ?? 0;
      const pct = Math.round(frac * 1000) / 10;
      const nTotal = tracks.n_total ?? summary?.release_songs;
      note.textContent =
        `${nSingle.toLocaleString("en-US")} songs (${pct}%) are single-stem` +
        (nTotal ? ` of ${Number(nTotal).toLocaleString("en-US")}` : "") +
        " — excluded from the bars above.";
    }
  }

  if (gm) {
    barChart("chart-gm", gm.labels.map(prettyLabel), gm.counts, {
      horizontal: true,
      color: PALETTE.bronze,
      compactTicks: true,
      valueLabel: "Songs",
    });
  }

  if (programs) {
    barChart("chart-programs", programs.labels, programs.counts, {
      horizontal: true,
      color: PALETTE.amberDeep,
      compactTicks: true,
      valueLabel: "Stems",
    });
  }

  if (backends && backends.labels?.length) {
    doughnutChart("chart-backends", backends.labels, backends.counts);
    const note = document.getElementById("backends-note");
    if (note) {
      note.textContent =
        backends.note ||
        "Soundfont stems: basic (single bank) vs varied (per-category pools); MIDI-DDSP is neural.";
    }
  }

  if (chunks) {
    barChart(
      "chart-chunks",
      chunks.chunk.map((c) => String(c)),
      chunks.n_songs,
      { color: PALETTE.marigoldSoft, compactTicks: true, valueLabel: "Songs" }
    );
  }

  if (duration) {
    const panel = document.getElementById("duration-panel");
    if (panel) {
      const p = duration.percentiles || {};
      const s = duration.summary || {};
      const cells = [
        ["Median", `${p.p50 ?? "—"} s`],
        ["p95", `${p.p95 ?? "—"} s`],
        ["≤120 s", `${s.pct_songs_under_120s ?? "—"}%`],
        ["≤380 s", `${s.pct_songs_under_380s ?? "—"}%`],
      ];
      panel.innerHTML = cells
        .map(
          ([label, value]) =>
            `<div><strong>${value}</strong><span>${label}</span></div>`
        )
        .join("");
    }
  }
}
