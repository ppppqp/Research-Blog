type Diagram = {
  source: string;
  codeBlock: HTMLPreElement;
  output: HTMLDivElement;
  error: HTMLParagraphElement;
  theme?: string;
};

const diagrams = new WeakMap<HTMLElement, Diagram>();
let mermaidPromise: Promise<(typeof import("mermaid"))["default"]> | undefined;
let revision = 0;
let rendering = false;
let pending = false;
let nextId = 0;

function collectDiagrams(): HTMLElement[] {
  document
    .querySelectorAll<HTMLElement>("pre > code.language-mermaid")
    .forEach(code => {
      if (code.closest(".mermaid-diagram")) return;

      const codeBlock = code.parentElement as HTMLPreElement;
      const container = document.createElement("div");
      container.className = "mermaid-diagram not-prose";
      container.tabIndex = 0;
      container.setAttribute("role", "group");
      container.setAttribute("aria-label", "Mermaid diagram");

      const output = document.createElement("div");
      const error = document.createElement("p");
      error.className = "mermaid-error";
      error.setAttribute("role", "status");
      error.textContent = "Unable to render diagram. Source is shown below.";
      error.hidden = true;

      codeBlock.replaceWith(container);
      container.append(output, error, codeBlock);
      diagrams.set(container, {
        source: code.textContent ?? "",
        codeBlock,
        output,
        error,
      });
    });

  return Array.from(
    document.querySelectorAll<HTMLElement>(".mermaid-diagram")
  ).filter(container => diagrams.has(container));
}

function showSource(diagram: Diagram): void {
  diagram.output.replaceChildren();
  diagram.codeBlock.hidden = false;
  diagram.error.hidden = false;
}

async function renderDiagrams(): Promise<void> {
  const containers = collectDiagrams();
  if (containers.length === 0) return;

  const currentRevision = revision;
  const theme =
    document.documentElement.dataset.theme === "dark" ? "dark" : "default";
  const unrendered = containers.filter(
    container => diagrams.get(container)?.theme !== theme
  );
  if (unrendered.length === 0) return;

  let mermaid: (typeof import("mermaid"))["default"];
  try {
    // Vite creates a separate chunk, fetched only on pages containing diagrams.
    mermaidPromise ??= import("mermaid").then(module => module.default);
    [mermaid] = await Promise.all([mermaidPromise, document.fonts.ready]);
  } catch {
    mermaidPromise = undefined;
    for (const container of unrendered) showSource(diagrams.get(container)!);
    return;
  }

  if (currentRevision !== revision) return;

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    suppressErrorRendering: true,
    theme,
    darkMode: theme === "dark",
    htmlLabels: false,
    fontFamily: "Arial, sans-serif",
  });

  for (const container of unrendered) {
    if (currentRevision !== revision) return;
    if (!container.isConnected) continue;
    const diagram = diagrams.get(container)!;

    // Render outside the visible diagram so theme changes never expose raw SVG
    // or replace a newer page's content when a navigation happens mid-render.
    const target = document.createElement("div");
    target.className = "mermaid-render-target";
    document.body.append(target);

    try {
      const { svg, bindFunctions } = await mermaid.render(
        `mermaid-diagram-${nextId++}`,
        diagram.source,
        target
      );
      if (currentRevision !== revision || !container.isConnected) continue;
      diagram.output.innerHTML = svg;
      const image = diagram.output.querySelector("svg");
      if (image?.viewBox.baseVal.width) {
        image.style.width = `${image.viewBox.baseVal.width}px`;
      }
      bindFunctions?.(diagram.output);
      diagram.codeBlock.hidden = true;
      diagram.error.hidden = true;
      diagram.theme = theme;
    } catch {
      if (currentRevision !== revision || !container.isConnected) continue;
      showSource(diagram);
      diagram.theme = theme;
    } finally {
      target.remove();
    }
  }
}

async function scheduleRender(): Promise<void> {
  revision++;
  pending = true;
  if (rendering) return;

  rendering = true;
  try {
    // Serialize Mermaid calls and coalesce navigation / theme changes.
    while (pending) {
      pending = false;
      await renderDiagrams();
    }
  } finally {
    rendering = false;
  }
}

document.addEventListener("astro:page-load", () => void scheduleRender());
new MutationObserver(() => void scheduleRender()).observe(
  document.documentElement,
  { attributes: true, attributeFilter: ["data-theme"] }
);
