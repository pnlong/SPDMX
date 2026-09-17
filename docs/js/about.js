import { mountChrome } from "./chrome.js";
import { renderCharts } from "./charts.js";

mountChrome({ active: "about" });
renderCharts().catch((err) => console.error(err));
