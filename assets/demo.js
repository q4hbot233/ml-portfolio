// Shared mount for the demos that run Python in the page.
//
// Streamlit's defaults are a white page in a light-mode browser, its own typeface, a
// Deploy button and a stack of start-up toasts, which reads as a different website from
// the dark landing page. The theme lives here so the demos cannot drift apart, and
// each page passes only the accent colour of the card that links to it.

import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@1.9.1/build/stlite.js";

const SITE = {
  background: "#08090B",
  panel: "#12151A",
  text: "#E9ECF1",
  border: "#1F232B",
};

// How long the loading screen may stay up before it gets out of the way regardless.
// Past three minutes something has gone wrong, and the reader is better served by
// Streamlit's own state than by a screen that never changes.
const VEIL_TIMEOUT_MS = 180_000;

// Anything Streamlit draws once the script is running. An exception counts: a page that
// failed should show the failure, not keep saying it is starting.
const DRAWN = [
  '[data-testid="stHeading"]',
  '[data-testid="stMarkdown"]',
  '[data-testid="stException"]',
  '[data-testid="stAlert"]',
].join(",");

function liftVeilWhenDrawn(root) {
  const veil = document.getElementById("veil");
  if (!veil) return;
  let lifted = false;
  const lift = () => {
    if (lifted) return;
    lifted = true;
    observer.disconnect();
    veil.classList.add("gone");
    setTimeout(() => veil.remove(), 450);
  };
  const observer = new MutationObserver(() => {
    if (root.querySelector(DRAWN)) lift();
  });
  observer.observe(root, { childList: true, subtree: true });
  setTimeout(lift, VEIL_TIMEOUT_MS);
}

export function mountDemo({ hue, requirements, files, entrypoint = "app.py" }) {
  document.documentElement.style.setProperty("--hue", hue);
  const root = document.getElementById("root");
  liftVeilWhenDrawn(root);
  return mount(
    {
      requirements,
      entrypoint,
      files,
      streamlitConfig: {
        "theme.base": "dark",
        "theme.primaryColor": hue,
        "theme.backgroundColor": SITE.background,
        "theme.secondaryBackgroundColor": SITE.panel,
        "theme.textColor": SITE.text,
        "theme.borderColor": SITE.border,
        "theme.font": "Schibsted Grotesk",
        "theme.codeFont": "Roboto Mono",
        "client.toolbarMode": "minimal",
      },
      disableProgressToasts: true,
      disableModuleAutoLoadToasts: true,
    },
    root,
  );
}
