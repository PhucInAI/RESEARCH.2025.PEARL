// PEARL project page: medium simulator, depth gauge, scene player, math.
(() => {
  "use strict";

  // Illustrative per-metre attenuation (R > G > B) and backscatter colour.
  // These are fixed demo values, not coefficients learned by PEARL.
  const BETA = [0.42, 0.11, 0.07];
  const VEIL = [0.02, 0.11, 0.17];
  const MAX_ADD = 9;      // metres of water added at the right end
  const MAX_REMOVE = 2.6; // metres of water removed at the left end
  const CHART_D = 10;     // x-range of the transmittance chart, metres
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- Math ----------
  function renderMath() {
    if (!window.katex) return;
    document.querySelectorAll("[data-tex]").forEach((el) => {
      katex.render(el.dataset.tex, el, {
        displayMode: el.classList.contains("eq"),
        throwOnError: false,
      });
    });
  }

  // ---------- Medium simulator ----------
  function initSimulator() {
    const canvas = document.getElementById("sim-canvas");
    const slider = document.getElementById("sim-slider");
    const readout = document.getElementById("sim-readout");
    if (!canvas || !slider) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const W = canvas.width, H = canvas.height;
    const img = new Image();
    img.src = "static/images/teaser.jpg";

    let lin, rel, out, pending = false;

    img.onload = () => {
      ctx.drawImage(img, 0, 0, W, H);
      const src = ctx.getImageData(0, 0, W, H);
      out = ctx.createImageData(W, H);
      lin = new Float32Array(W * H * 3);
      rel = new Float32Array(W * H);
      for (let y = 0; y < H; y++) {
        // Pseudo-depth: the top of this frame is the far reef.
        const fy = 0.35 + 0.65 * Math.pow(1 - y / H, 1.3);
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          rel[i] = fy * (1 + 0.15 * x / W);
          for (let c = 0; c < 3; c++) lin[i * 3 + c] = Math.pow(src.data[i * 4 + c] / 255, 2.2);
        }
      }
      for (let i = 3; i < out.data.length; i += 4) out.data[i] = 255;
      draw();
    };

    function draw() {
      pending = false;
      if (!lin) return;
      const s = Number(slider.value) / 100;
      const add = s >= 0;
      const D = add ? s * MAX_ADD : -s * MAX_REMOVE;
      const N = W * H, o = out.data;

      // Exposure for the restored image: a sparse high percentile.
      let gain = 1;
      if (!add && D > 0) {
        const samples = [];
        for (let i = 0; i < N; i += 53) {
          for (let c = 0; c < 3; c++) {
            const T = Math.exp(-BETA[c] * rel[i] * D);
            samples.push(Math.max(0, (lin[i * 3 + c] - VEIL[c] * (1 - T)) / Math.max(T, 1e-3)));
          }
        }
        samples.sort((a, b) => a - b);
        const p = samples[Math.floor(samples.length * 0.995)];
        gain = p > 0.9 ? 0.9 / p : 1;
      }

      for (let i = 0; i < N; i++) {
        const d = rel[i] * D;
        for (let c = 0; c < 3; c++) {
          const T = Math.exp(-BETA[c] * d);
          const v = lin[i * 3 + c];
          let r = add ? v * T + VEIL[c] * (1 - T)
                      : Math.max(0, (v - VEIL[c] * (1 - T)) / Math.max(T, 1e-3)) * gain;
          r = r > 1 ? 1 : r;
          o[i * 4 + c] = 255 * Math.pow(r, 1 / 2.2);
        }
      }
      ctx.putImageData(out, 0, 0);

      readout.textContent = D < 0.05
        ? "The original photograph"
        : add
          ? `${D.toFixed(1)} m more water: red fades first, then green, and the veil rises`
          : `${D.toFixed(1)} m of water inverted out of the photo: the reef's own colours return`;
      updateChart(add ? D : 0);
    }

    slider.addEventListener("input", () => {
      if (!pending) { pending = true; requestAnimationFrame(draw); }
    });
    buildChart();
  }

  // ---------- Transmittance chart ----------
  const CX0 = 34, CX1 = 310, CY0 = 16, CY1 = 160;
  const sx = (d) => CX0 + (d / CHART_D) * (CX1 - CX0);
  const sy = (t) => CY1 - t * (CY1 - CY0);

  function buildChart() {
    ["r", "g", "b"].forEach((ch, c) => {
      let p = "";
      for (let k = 0; k <= 60; k++) {
        const d = (k / 60) * CHART_D;
        p += (k ? "L" : "M") + sx(d).toFixed(1) + " " + sy(Math.exp(-BETA[c] * d)).toFixed(1);
      }
      document.getElementById("curve-" + ch).setAttribute("d", p);
    });
    updateChart(0);
  }

  function updateChart(D) {
    const x = sx(Math.min(D, CHART_D));
    const mark = document.getElementById("chart-mark");
    if (!mark) return;
    mark.setAttribute("x1", x); mark.setAttribute("x2", x);
    ["r", "g", "b"].forEach((ch, c) => {
      const dot = document.getElementById("dot-" + ch);
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", sy(Math.exp(-BETA[c] * D)));
    });
  }

  // ---------- Depth gauge ----------
  function initGauge() {
    const gauge = document.querySelector(".gauge");
    if (!gauge) return;
    const meters = document.getElementById("gauge-m");
    const bars = gauge.querySelectorAll(".gauge-bar i");
    let ticking = false;
    function update() {
      ticking = false;
      const max = document.documentElement.scrollHeight - innerHeight;
      const f = max > 0 ? scrollY / max : 0;
      const d = f * 12;
      gauge.classList.toggle("on", scrollY > innerHeight * 0.6);
      meters.textContent = d.toFixed(1);
      bars.forEach((b, c) => { b.style.transform = `scaleX(${Math.exp(-BETA[c] * d)})`; });
    }
    addEventListener("scroll", () => {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  // ---------- Scene player ----------
  function initPlayer() {
    const video = document.getElementById("scene-video");
    const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
    if (!video || !tabs.length) return;

    function select(tab) {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", on);
        t.tabIndex = on ? 0 : -1;
      });
      video.src = tab.dataset.src;
      video.play().catch(() => {});
    }
    tabs.forEach((t, i) => {
      t.tabIndex = i ? -1 : 0;
      t.addEventListener("click", () => select(t));
      t.addEventListener("keydown", (e) => {
        const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!step) return;
        const next = tabs[(i + step + tabs.length) % tabs.length];
        next.focus(); select(next);
      });
    });

    // Play only while on screen.
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !reduceMotion) video.play().catch(() => {});
      else video.pause();
    }, { threshold: 0.35 }).observe(video);

    const hero = document.querySelector(".hero-video");
    if (hero && reduceMotion) hero.pause();
  }

  // ---------- Copy BibTeX ----------
  function initCopy() {
    document.querySelectorAll("[data-copy]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = document.getElementById(btn.dataset.copy).innerText;
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = "Copied";
        } catch {
          btn.textContent = "Select and copy";
        }
        setTimeout(() => { btn.textContent = "Copy BibTeX"; }, 1800);
      });
    });
  }

  renderMath();
  initSimulator();
  initGauge();
  initPlayer();
  initCopy();
})();
