import { mountChrome } from "./chrome.js";
import { renderSoundfonts } from "./soundfonts.js";

mountChrome({ active: "soundfonts" });
renderSoundfonts().catch((err) => console.error(err));
