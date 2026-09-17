import { mountChrome } from "./chrome.js";
import { renderHomeStats } from "./charts.js";

/** Legacy entry; prefer page-specific modules (home.js, about.js, …). */
mountChrome({ active: "home" });
renderHomeStats().catch((err) => console.error(err));
