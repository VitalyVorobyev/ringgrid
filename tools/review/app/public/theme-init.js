// Set the theme class before the first paint (in agreement with @vitavision/ui's theme.ts).
// A file rather than an inline script, so the server's CSP allows scripts from 'self' only.
(function () {
  try {
    var stored = localStorage.getItem("vitavision-theme");
    var dark = stored === "dark" || ((stored === null || stored === "system") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    if (dark) document.documentElement.classList.add("dark");
  } catch (_) {
    // Storage unavailable: the light palette on :root applies.
  }
})();
