(function () {
  "use strict";

  const PRESETS = {
    electric: {
      name: "Electric Gaze",
      renderMode: "dither",
      bgMode: "none",
      bgBlur: 12,
      bgOpacity: 90,
      cellSize: 9,
      coverage: 100,
      invert: false,
      styleBlend: "source-over",
      charSet: "standard",
      customChars: "",
      brightness: 0,
      contrast: 158,
      edgeEmphasis: 0,
      density: 20,
      toneCurve: [[0, 0], [1, 1]],
      tint: "#3ca6ff",
      tintOpacity: 0,
      overlayBlend: "multiply",
      saturation: 100,
      grayscale: 0,
      blurType: "off",
      blurAmount: 35,
      animated: true,
      animStyle: "shimmer",
      animSpeed: { enabled: true, intensity: 100 },
      animIntensity: { enabled: true, intensity: 60 },
      lights: { enabled: false, points: [] },
      mask: { enabled: false, invert: false, dataUrl: null },
      pfx: {
        vignette: { enabled: false, intensity: 38 },
        scanLines: { enabled: false, intensity: 40 },
        chromatic: { enabled: false, intensity: 15 },
        bloom: { enabled: false, intensity: 25 },
        filmGrain: { enabled: false, intensity: 30 },
        glitch: { enabled: false, intensity: 20 },
        halftone: { enabled: false, intensity: 20 },
        pixelate: { enabled: false, intensity: 15 },
        filmDust: { enabled: false, intensity: 20 }
      }
    },
    vignette: {
      name: "Vignette Bloom",
      renderMode: "mosaic",
      bgMode: "solid",
      bgBlur: 12,
      bgOpacity: 90,
      cellSize: 16,
      coverage: 100,
      invert: false,
      styleBlend: "source-over",
      charSet: "standard",
      customChars: "",
      brightness: 12,
      contrast: 115,
      edgeEmphasis: 0,
      density: 0,
      toneCurve: [[0, 0], [1, 1]],
      tint: "#3ca6ff",
      tintOpacity: 0,
      overlayBlend: "multiply",
      saturation: 100,
      grayscale: 0,
      blurType: "off",
      blurAmount: 35,
      animated: true,
      animStyle: "wave",
      animSpeed: { enabled: true, intensity: 100 },
      animIntensity: { enabled: true, intensity: 60 },
      lights: { enabled: false, points: [] },
      mask: { enabled: false, invert: false, dataUrl: null },
      pfx: {
        vignette: { enabled: true, intensity: 38 },
        scanLines: { enabled: false, intensity: 40 },
        chromatic: { enabled: false, intensity: 15 },
        bloom: { enabled: true, intensity: 25 },
        filmGrain: { enabled: false, intensity: 30 },
        glitch: { enabled: false, intensity: 20 },
        halftone: { enabled: false, intensity: 20 },
        pixelate: { enabled: false, intensity: 15 },
        filmDust: { enabled: false, intensity: 20 }
      }
    },
    ink: {
      name: "Ink Garden",
      renderMode: "dither",
      bgMode: "none",
      bgBlur: 12,
      bgOpacity: 90,
      cellSize: 9,
      coverage: 100,
      invert: false,
      styleBlend: "source-over",
      charSet: "standard",
      customChars: "",
      brightness: 0,
      contrast: 158,
      edgeEmphasis: 0,
      density: 20,
      toneCurve: [[0, 0], [1, 1]],
      tint: "#3ca6ff",
      tintOpacity: 0,
      overlayBlend: "multiply",
      saturation: 100,
      grayscale: 0,
      blurType: "off",
      blurAmount: 35,
      animated: true,
      animStyle: "pulse",
      animSpeed: { enabled: true, intensity: 100 },
      animIntensity: { enabled: true, intensity: 60 },
      lights: { enabled: false, points: [] },
      mask: { enabled: false, invert: true, dataUrl: null, showOverlay: true },
      pfx: {
        vignette: { enabled: false, intensity: 38 },
        scanLines: { enabled: false, intensity: 40 },
        chromatic: { enabled: false, intensity: 15 },
        bloom: { enabled: false, intensity: 25 },
        filmGrain: { enabled: false, intensity: 30 },
        glitch: { enabled: false, intensity: 20 },
        halftone: { enabled: false, intensity: 20 },
        pixelate: { enabled: false, intensity: 15 },
        filmDust: { enabled: false, intensity: 20 }
      }
    }
  };

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function hash(x, y) {
    let h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
    return h - Math.floor(h);
  }
  function hexRgb(hex) {
    const h = String(hex || "#000000").replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(r, g, b, a) { return `rgba(${r | 0},${g | 0},${b | 0},${clamp(a, 0, 1)})`; }
  function mergeDeep(dst, src) {
    for (const k in src) {
      if (src[k] && typeof src[k] === "object" && !Array.isArray(src[k]) && dst[k] && typeof dst[k] === "object") mergeDeep(dst[k], src[k]);
      else dst[k] = src[k];
    }
    return dst;
  }
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
  function drawCover(ctx, img, w, h) {
    const s = Math.max(w / img.width, h / img.height);
    const iw = img.width * s, ih = img.height * s;
    ctx.drawImage(img, (w - iw) / 2, (h - ih) / 2, iw, ih);
  }

  function create(canvas, options) {
    options = options || {};
    const preset = typeof options.preset === "string" ? PRESETS[options.preset] : (options.preset || PRESETS.electric);
    const params = mergeDeep(mergeDeep({}, preset), options.params || {});
    if (options.theme) params.theme = options.theme;
    params.pfx = params.pfx || {};
    params.lights = params.lights || { enabled: false, points: [] };
    params.mask = params.mask || { enabled: false, invert: false, dataUrl: null };

    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    let img = null, maskImg = null, cells = [], cols = 0, rows = 0, cw = 0, ch = 0;
    let raf = 0, running = false, destroyed = false, snap = null, snapCtx = null;
    const sample = document.createElement("canvas");
    const sampleCtx = sample.getContext("2d", { willReadFrequently: true });

    function tone(v) {
      const pts = params.toneCurve || [[0, 0], [1, 1]];
      if (pts.length < 2) return clamp(v, 0, 1);
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        if (v <= b.x) return lerp(a.y, b.y, (v - a.x) / ((b.x - a.x) || 1));
      }
      return pts[pts.length - 1].y;
    }
    function blendChannel(c, t, mode) {
      if (mode === "screen") return 1 - (1 - c) * (1 - t);
      if (mode === "overlay") return c < 0.5 ? 2 * c * t : 1 - 2 * (1 - c) * (1 - t);
      return c * t; // multiply default
    }
    function adjustColor(r, g, b) {
      let rr = r / 255, gg = g / 255, bb = b / 255;
      const lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb;
      const bf = 1 + (params.brightness || 0) / 100;
      const cf = 1 + ((params.contrast ?? 100) - 100) / 100;
      rr = clamp((rr * bf - 0.5) * cf + 0.5, 0, 1);
      gg = clamp((gg * bf - 0.5) * cf + 0.5, 0, 1);
      bb = clamp((bb * bf - 0.5) * cf + 0.5, 0, 1);
      const gray = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb;
      const sat = (params.saturation ?? 100) / 100;
      rr = gray + (rr - gray) * sat; gg = gray + (gg - gray) * sat; bb = gray + (bb - gray) * sat;
      const grayMix = (params.grayscale || 0) / 100;
      rr = lerp(rr, gray, grayMix); gg = lerp(gg, gray, grayMix); bb = lerp(bb, gray, grayMix);
      rr = tone(rr); gg = tone(gg); bb = tone(bb);
      const tintAmt = (params.tintOpacity || 0) / 100;
      if (tintAmt > 0 && params.tint) {
        const [tr, tg, tb] = hexRgb(params.tint).map(v => v / 255);
        const mode = params.overlayBlend || "multiply";
        rr = lerp(rr, blendChannel(rr, tr, mode), tintAmt);
        gg = lerp(gg, blendChannel(gg, tg, mode), tintAmt);
        bb = lerp(bb, blendChannel(bb, tb, mode), tintAmt);
      }
      return [rr * 255, gg * 255, bb * 255, lum];
    }

    function buildCells() {
      if (!img) return;
      cols = Math.max(8, Math.floor(W / Math.max(4, params.cellSize || 10)));
      rows = Math.max(8, Math.floor(H / Math.max(4, params.cellSize || 10)));
      cw = W / cols; ch = H / rows;
      sample.width = cols; sample.height = rows;
      sampleCtx.clearRect(0, 0, cols, rows);
      drawCover(sampleCtx, img, cols, rows);
      const data = sampleCtx.getImageData(0, 0, cols, rows).data;
      cells = [];
      const lums = [];
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const i = (y * cols + x) * 4;
          const [r, g, b, lum] = adjustColor(data[i], data[i + 1], data[i + 2], data[i + 3]);
          const cell = { x: x * cw + cw / 2, y: y * ch + ch / 2, w: cw, h: ch, r, g, b, lum, cx: x, cy: y, edge: 0 };
          cells.push(cell); lums.push(lum);
        }
      }
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const c = cells[y * cols + x];
        let sum = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= cols || yy >= rows) continue;
          sum += lums[yy * cols + xx]; n++;
        }
        c.edge = Math.abs(c.lum - sum / n);
      }
    }

    function motionFor(c, t) {
      if (!params.animated) return 0.5;
      const speed = (params.animSpeed && params.animSpeed.enabled ? params.animSpeed.intensity : 0) / 100;
      const inten = (params.animIntensity && params.animIntensity.enabled ? params.animIntensity.intensity : 0) / 100;
      const style = params.animStyle || "pulse";
      const nx = c.cx / Math.max(1, cols - 1), ny = c.cy / Math.max(1, rows - 1);
      let raw = 0.5;
      if (style === "wave") raw = 0.5 + 0.5 * Math.sin(nx * 6.283 + t * speed * 4);
      else if (style === "pulse") raw = 0.5 + 0.5 * Math.sin(t * speed * 5 + ny * 2);
      else if (style === "shimmer") raw = 0.72 + 0.28 * Math.sin((c.cx + c.cy) * 0.55 + t * speed * 8);
      else if (style === "ripple") raw = 0.5 + 0.5 * Math.sin(Math.hypot(c.cx - cols / 2, c.cy - rows / 2) * 0.32 - t * speed * 6);
      else if (style === "flicker") raw = 0.62 + 0.38 * Math.sin(t * speed * 18 + hash(c.cx, c.cy) * 6.283);
      return 0.5 + (raw - 0.5) * inten;
    }

    function prim(c, t, motion) {
      const s = Math.min(c.w, c.h);
      const lum = params.invert ? 1 - c.lum : c.lum;
      const sizeMul = 1 + (motion - 0.5) * 0.7 * ((params.animIntensity && params.animIntensity.intensity) || 0) / 100;
      const alpha = 0.62 + 0.38 * motion;
      const edgeBoost = 1 + (params.edgeEmphasis || 0) / 100 * c.edge * 8;
      const fill = rgba(c.r, c.g, c.b, alpha);
      const stroke = rgba(c.r, c.g, c.b, alpha * 0.9);
      const mode = params.renderMode || "dither";
      ctx.fillStyle = fill; ctx.strokeStyle = stroke;
      if (params.styleBlend && ctx.globalCompositeOperation !== params.styleBlend) ctx.globalCompositeOperation = params.styleBlend;

      if (mode === "characters" || mode === "hexdump" || mode === "matrix") {
        const std = " .:-=+*#%@";
        const custom = params.customChars || "";
        const chars = mode === "hexdump" ? "0123456789ABCDEF" : (custom || (params.charSet === "blocks" ? " ░▒▓█" : std));
        let idx = Math.floor(clamp(lum, 0, 0.999) * (chars.length - 1));
        if (mode === "matrix") {
          const rain = "アカサタナハマヤラワ0123456789$#";
          idx = Math.floor(hash(c.cx, c.cy + Math.floor(t * 7)) * (rain.length - 1));
          ctx.fillStyle = rgba(40, 255, 90, 0.28 + 0.5 * lum * motion);
          ctx.font = `${Math.max(8, c.h * 0.95)}px ui-monospace, monospace`;
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(rain[idx], c.x, c.y + (motion - 0.5) * c.h * 0.5);
          return;
        }
        ctx.fillStyle = fill;
        ctx.font = `${Math.max(8, c.h * 0.95)}px ui-monospace, monospace`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(chars[idx], c.x, c.y);
        return;
      }
      if (mode === "halfblocks") {
        const top = lum, bottom = clamp(lum * 0.55 + (hash(c.cx, c.cy) - 0.5) * 0.2, 0, 1);
        ctx.fillStyle = rgba(c.r, c.g, c.b, alpha * top);
        ctx.fillRect(c.x - c.w / 2, c.y - c.h / 2, c.w, c.h / 2);
        ctx.fillStyle = rgba(c.r * 0.55, c.g * 0.55, c.b * 0.65, alpha * bottom);
        ctx.fillRect(c.x - c.w / 2, c.y, c.w, c.h / 2);
        return;
      }
      if (mode === "braille") {
        const bits = Math.floor(clamp(lum, 0, 0.999) * 255);
        const dots = [[-0.22, -0.36], [0.22, -0.36], [-0.22, -0.12], [0.22, -0.12], [-0.22, 0.12], [0.22, 0.12], [-0.22, 0.36], [0.22, 0.36]];
        ctx.fillStyle = fill;
        for (let i = 0; i < 8; i++) if (bits & (1 << i)) {
          ctx.beginPath(); ctx.arc(c.x + dots[i][0] * c.w, c.y + dots[i][1] * c.h, Math.max(0.8, c.w * 0.055), 0, Math.PI * 2); ctx.fill();
        }
        return;
      }

      const den = (params.density ?? 0) / 100;
      const jitter = (hash(c.cx, c.cy) - 0.5) * den * s * 0.35;
      const x = c.x + (motion - 0.5) * c.w * 0.22 + jitter;
      const y = c.y + (motion - 0.5) * c.h * 0.22;
      const w = c.w * sizeMul * edgeBoost, h = c.h * sizeMul * edgeBoost;

      switch (mode) {
        case "mosaic":
          ctx.fillRect(x - w / 2, y - h / 2, w, h); break;
        case "pixel":
          ctx.fillRect(x - w * lum / 2, y - h * lum / 2, w * Math.max(0.12, lum), h * Math.max(0.12, lum)); break;
        case "dots": {
          ctx.beginPath(); ctx.arc(x, y, Math.max(0.6, s * 0.42 * lum * (0.5 + den)), 0, Math.PI * 2); ctx.fill(); break;
        }
        case "cross": {
          ctx.lineWidth = Math.max(1, s * 0.07); ctx.beginPath();
          ctx.moveTo(x - w * lum / 2, y - h * lum / 2); ctx.lineTo(x + w * lum / 2, y + h * lum / 2);
          ctx.moveTo(x + w * lum / 2, y - h * lum / 2); ctx.lineTo(x - w * lum / 2, y + h * lum / 2); ctx.stroke(); break;
        }
        case "diamond": {
          ctx.beginPath(); ctx.moveTo(x, y - h * lum / 2); ctx.lineTo(x + w * lum / 2, y); ctx.lineTo(x, y + h * lum / 2); ctx.lineTo(x - w * lum / 2, y); ctx.closePath(); ctx.fill(); break;
        }
        case "lines": {
          ctx.lineWidth = Math.max(1, s * 0.08); ctx.beginPath();
          const yy = y + (hash(c.cx, c.cy) - 0.5) * h * den;
          ctx.moveTo(x - w / 2, yy); ctx.lineTo(x - w / 2 + w * lum, yy); ctx.stroke(); break;
        }
        case "diagonal": {
          ctx.lineWidth = Math.max(1, s * 0.07); ctx.beginPath();
          ctx.moveTo(x - w / 2, y + h / 2); ctx.lineTo(x + w / 2, y - h / 2); ctx.stroke(); break;
        }
        case "rings": {
          ctx.lineWidth = Math.max(1, s * 0.06); ctx.beginPath(); ctx.arc(x, y, Math.max(1, s * 0.48 * lum), 0, Math.PI * 2); ctx.stroke(); break;
        }
        case "hearts": {
          if (lum < 0.12) break;
          const k = s * 0.55 * lum;
          ctx.beginPath();
          ctx.moveTo(x, y + k * 0.75);
          ctx.bezierCurveTo(x - k * 1.4, y - k * 0.1, x - k * 0.7, y - k * 1.2, x, y - k * 0.35);
          ctx.bezierCurveTo(x + k * 0.7, y - k * 1.2, x + k * 1.4, y - k * 0.1, x, y + k * 0.75);
          ctx.fill(); break;
        }
        case "stars": {
          const k = s * 0.5 * lum, spikes = 4;
          ctx.beginPath();
          for (let i = 0; i < spikes * 2; i++) {
            const r = i % 2 ? k * 0.38 : k, a = (Math.PI / spikes) * i - Math.PI / 2;
            const px = x + Math.cos(a) * r, py = y + Math.sin(a) * r;
            i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.closePath(); ctx.fill(); break;
        }
        case "hexagons": {
          const k = s * 0.5;
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const a = Math.PI / 6 + i * Math.PI / 3, px = x + Math.cos(a) * k, py = y + Math.sin(a) * k;
            i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.closePath(); ctx.globalAlpha = alpha * (0.25 + lum * 0.75); ctx.fill(); ctx.globalAlpha = 1; break;
        }
        case "triangles": {
          const k = s * 0.55 * (0.35 + lum * 0.65);
          ctx.beginPath(); ctx.moveTo(x, y - k); ctx.lineTo(x + k * 0.9, y + k * 0.7); ctx.lineTo(x - k * 0.9, y + k * 0.7); ctx.closePath(); ctx.fill(); break;
        }
        case "bubbles": {
          ctx.globalAlpha = alpha * (0.25 + lum * 0.55); ctx.beginPath(); ctx.arc(x, y, Math.max(1, s * 0.48), 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1; break;
        }
        case "hatch": {
          ctx.lineWidth = Math.max(1, s * 0.05);
          const n = Math.floor(1 + lum * 4 + den * 3);
          for (let i = 0; i < n; i++) {
            const off = (i - n / 2) * s * 0.18;
            ctx.beginPath(); ctx.moveTo(x - w / 2 + off, y + h / 2); ctx.lineTo(x + w / 2 + off, y - h / 2); ctx.stroke();
          } break;
        }
        case "contour": {
          const band = (lum * 7) % 1;
          if (band < 0.16) { ctx.lineWidth = Math.max(1, s * 0.05); ctx.beginPath(); ctx.arc(x, y, s * 0.42, 0, Math.PI * 2); ctx.stroke(); }
          break;
        }
        case "voxel": case "lego": {
          const k = s * 0.42 * (0.45 + lum * 0.55);
          ctx.fillStyle = rgba(Math.min(255, c.r * 1.15), Math.min(255, c.g * 1.15), Math.min(255, c.b * 1.15), alpha);
          ctx.beginPath(); ctx.moveTo(x - k / 2, y - k / 2); ctx.lineTo(x + k / 2, y - k / 2); ctx.lineTo(x, y - k); ctx.lineTo(x - k, y - k / 2); ctx.closePath(); ctx.fill();
          ctx.fillStyle = rgba(c.r * 0.72, c.g * 0.72, c.b * 0.8, alpha);
          ctx.beginPath(); ctx.moveTo(x - k, y - k / 2); ctx.lineTo(x - k / 2, y + k / 2); ctx.lineTo(x, y); ctx.lineTo(x - k / 2, y - k / 2); ctx.closePath(); ctx.fill();
          ctx.fillStyle = rgba(c.r * 0.5, c.g * 0.5, c.b * 0.62, alpha);
          ctx.beginPath(); ctx.moveTo(x - k / 2, y + k / 2); ctx.lineTo(x + k / 2, y + k / 2); ctx.lineTo(x + k, y); ctx.lineTo(x, y); ctx.closePath(); ctx.fill();
          break;
        }
        case "mixed": {
          const pick = (c.cx + c.cy) % 5;
          if (pick === 0) { ctx.beginPath(); ctx.arc(x, y, Math.max(1, s * 0.35 * lum), 0, Math.PI * 2); ctx.fill(); }
          else if (pick === 1) { ctx.fillRect(x - w / 2, y - h / 2, w * lum, h); }
          else if (pick === 2) { ctx.beginPath(); ctx.moveTo(x, y - h / 2); ctx.lineTo(x + w / 2, y + h / 2); ctx.lineTo(x - w / 2, y + h / 2); ctx.closePath(); ctx.fill(); }
          else if (pick === 3) { ctx.lineWidth = Math.max(1, s * 0.06); ctx.beginPath(); ctx.moveTo(x - w / 2, y); ctx.lineTo(x + w / 2, y); ctx.stroke(); }
          else { ctx.beginPath(); ctx.arc(x, y, Math.max(1, s * 0.4), 0, Math.PI * 2); ctx.stroke(); }
          break;
        }
        case "disco": {
          const hue = (t * 55 + c.cx * 4 + c.cy * 6) % 360;
          ctx.fillStyle = `hsla(${hue},85%,${20 + lum * 45}%,${alpha})`;
          ctx.beginPath(); ctx.arc(x, y, Math.max(1, s * 0.4 * lum), 0, Math.PI * 2); ctx.fill(); break;
        }
        case "dither": default: {
          const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
          const threshold = bayer[(c.cy % 4) * 4 + (c.cx % 4)] / 16 + den * 0.35;
          if (lum > threshold) ctx.fillRect(x - w / 2, y - h / 2, w, h);
          break;
        }
      }
      if (params.styleBlend && ctx.globalCompositeOperation !== "source-over") ctx.globalCompositeOperation = "source-over";
    }

    function drawBackground(t) {
      const themeBg = (params.theme && params.theme.bg) || "#070510";
      if (params.bgMode === "solid") {
        ctx.fillStyle = (params.theme && params.theme.bg) || "#0b0d15"; ctx.fillRect(0, 0, W, H); return;
      }
      ctx.fillStyle = themeBg; ctx.fillRect(0, 0, W, H);
      if (!img) return;
      if (params.bgMode === "original") {
        ctx.globalAlpha = (params.bgOpacity ?? 90) / 100; drawCover(ctx, img, W, H); ctx.globalAlpha = 1;
      } else if (params.bgMode === "blurred" || params.bgMode === "blur") {
        ctx.save(); ctx.globalAlpha = (params.bgOpacity ?? 90) / 100; ctx.filter = `blur(${params.bgBlur || 12}px)`; drawCover(ctx, img, W, H); ctx.restore();
      }
    }

    function ensureSnap() {
      if (!snap) { snap = document.createElement("canvas"); snap.width = W; snap.height = H; snapCtx = snap.getContext("2d"); }
      snapCtx.clearRect(0, 0, W, H); snapCtx.drawImage(canvas, 0, 0);
      return snap;
    }
    function applyPfx(t) {
      const p = params.pfx || {}; let need = false;
      for (const k in p) if (p[k] && p[k].enabled) { need = true; break; }
      if (!need) return;
      const src = ensureSnap();
      for (const key in p) {
        const fx = p[key]; if (!fx || !fx.enabled) continue;
        const a = (fx.intensity ?? 50) / 100;
        if (key === "vignette") {
          const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
          g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, `rgba(0,0,0,${0.15 + a * 0.55})`);
          ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
        } else if (key === "scanLines") {
          ctx.fillStyle = `rgba(255,255,255,${a * 0.08})`;
          for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
        } else if (key === "bloom") {
          ctx.save(); ctx.globalCompositeOperation = "screen"; ctx.globalAlpha = a * 0.55; ctx.filter = `blur(${6 + a * 10}px)`; ctx.drawImage(src, 0, 0); ctx.restore();
        } else if (key === "chromatic") {
          ctx.save(); ctx.globalCompositeOperation = "screen"; ctx.globalAlpha = a * 0.22;
          ctx.filter = "saturate(2.4) hue-rotate(18deg)"; ctx.drawImage(src, 2 + a * 4, 0);
          ctx.filter = "saturate(2.4) hue-rotate(-18deg)"; ctx.drawImage(src, -2 - a * 4, 0); ctx.restore();
        } else if (key === "filmGrain") {
          for (let i = 0; i < 260 * a; i++) { ctx.fillStyle = `rgba(${Math.random() > 0.5 ? 255 : 0},${Math.random() > 0.5 ? 255 : 0},255,${Math.random() * a * 0.16})`; ctx.fillRect(Math.random() * W, Math.random() * H, 1.5, 1.5); }
        } else if (key === "glitch") {
          if (Math.random() < a * 0.35) for (let i = 0; i < 5; i++) {
            const y = Math.random() * H, h = 2 + Math.random() * 7 * a, dx = (Math.random() - 0.5) * 28 * a;
            ctx.drawImage(src, 0, y, W, h, dx, y, W, h);
          }
        } else if (key === "halftone") {
          ctx.fillStyle = `rgba(255,255,255,${a * 0.12})`;
          const step = 7;
          for (let y = step / 2; y < H; y += step) for (let x = step / 2; x < W; x += step) { ctx.beginPath(); ctx.arc(x, y, a * 1.4, 0, Math.PI * 2); ctx.fill(); }
        } else if (key === "pixelate") {
          const size = Math.max(2, Math.round(a * 14));
          const tmp = document.createElement("canvas"); tmp.width = Math.max(1, W / size | 0); tmp.height = Math.max(1, H / size | 0);
          const tctx = tmp.getContext("2d"); tctx.imageSmoothingEnabled = false; tctx.drawImage(src, 0, 0, tmp.width, tmp.height);
          ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, W, H); ctx.restore();
        } else if (key === "filmDust") {
          for (let i = 0; i < 90 * a; i++) { ctx.fillStyle = `rgba(255,240,210,${Math.random() * a * 0.5})`; ctx.beginPath(); ctx.arc(Math.random() * W, Math.random() * H, Math.random() * 1.8 + 0.4, 0, Math.PI * 2); ctx.fill(); }
        }
      }
    }
    function drawLights() {
      const lights = params.lights; if (!lights || !lights.enabled) return;
      for (const p of lights.points || []) {
        const x = (p.x ?? 0.5) * W, y = (p.y ?? 0.5) * H, r = (p.radius ?? 0.2) * Math.max(W, H), a = (p.intensity ?? 0.5);
        const g = ctx.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, `rgba(255,220,160,${a})`); g.addColorStop(1, "rgba(255,220,160,0)");
        ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.fillStyle = g; ctx.fillRect(x - r, y - r, r * 2, r * 2); ctx.restore();
      }
    }
    function drawMask() {
      const m = params.mask; if (!m || !m.enabled || !maskImg) return;
      const tmp = document.createElement("canvas"); tmp.width = W; tmp.height = H; const tc = tmp.getContext("2d");
      tc.drawImage(img, 0, 0, W, H);
      tc.globalCompositeOperation = "destination-in";
      if (m.invert) { tc.filter = "invert(1)"; }
      drawCover(tc, maskImg, W, H);
      tc.filter = "none"; tc.globalCompositeOperation = "source-over";
      ctx.drawImage(tmp, 0, 0);
    }

    function render(t) {
      if (destroyed) return;
      drawBackground(t);
      const blend = params.styleBlend && ["source-over", "lighter", "screen", "multiply", "overlay"].includes(params.styleBlend) ? params.styleBlend : "source-over";
      ctx.globalCompositeOperation = blend;
      if (params.charSet === "blocks" || params.renderMode === "characters" || params.renderMode === "hexdump" || params.renderMode === "matrix" || params.renderMode === "halfblocks") {
        ctx.font = `${Math.max(8, ch * 0.95)}px ui-monospace, monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      }
      for (const c of cells) {
        if ((params.coverage ?? 100) < 100 && hash(c.cx * 7 + 1, c.cy * 13 + 5) * 100 > params.coverage) continue;
        const motion = motionFor(c, t);
        prim(c, t, motion);
      }
      ctx.globalCompositeOperation = "source-over";
      applyPfx(t); drawLights(); drawMask();
    }

    function tick(now) {
      if (destroyed || !running) return;
      render(now / 1000); raf = requestAnimationFrame(tick);
    }
    function start() { if (!running && !destroyed) { running = true; raf = requestAnimationFrame(tick); } }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }

    async function setImage(src) {
      img = await loadImage(src); buildCells(); start();
    }
    async function setMask(src) {
      if (!src) { maskImg = null; return; }
      maskImg = await loadImage(src);
    }
    if (options.image) {
      setImage(options.image).catch(() => { if (!destroyed) { cells = []; buildCells(); start(); } });
    } else { buildCells(); start(); }
    if (params.mask && params.mask.dataUrl) setMask(params.mask.dataUrl).catch(() => {});

    return {
      params, preset,
      setPreset(name) { const p = PRESETS[name]; if (!p) return; mergeDeep(params, p); buildCells(); },
      setImage, setMask,
      renderOnce() { render(performance.now() / 1000); },
      start, stop,
      destroy() { destroyed = true; stop(); }
    };
  }

  window.EmberAscii = { create, presets: PRESETS };
})();
