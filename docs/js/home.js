import { mountChrome } from "./chrome.js";
import { renderHomeStats } from "./charts.js";

mountChrome({ active: "home" });
renderHomeStats().catch((err) => console.error(err));
