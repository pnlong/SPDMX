import { mountChrome } from "./chrome.js";
import { renderCharts } from "./charts.js";
import { renderRecipeTable } from "./soundfonts.js";

mountChrome({ active: "about" });
renderCharts().catch((err) => console.error(err));
renderRecipeTable().catch((err) => console.error(err));
