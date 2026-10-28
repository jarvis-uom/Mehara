// Builds the proposal talk deck: node build_deck.js
// Pictures and data come from p1_maps.py, p2_images.py, p4_chart_data.py and p5_cards.py.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");
// optional helper that writes the theme colours into the file; the deck builds without it
let applyTheme = null;
try { ({ applyTheme } = require("./apply_theme.js")); } catch (e) { /* not present */ }

const ROOT = path.resolve(__dirname, "..");
const IMG = path.join(ROOT, "img");
const CARDS = path.join(IMG, "cards");
const OUT = path.join(ROOT, "SWOT Tank Water Tracker - proposal talk.pptx");
const STUDY_MAP = path.resolve(ROOT, "..", "proposal-drafts", "study-area-map.png");
const EXPLORER = "https://mibrechb.github.io/swot-data-explorer/";
const levels = require(path.join(ROOT, "data", "levels.json"));
const heights = require(path.join(ROOT, "data", "swot_heights_meta.json"));
const pics = require(path.join(ROOT, "data", "images_meta.json"));

const THEME = {
  name: "SWOT Tank Tracker",
  headFontFace: "Segoe UI Semibold",
  bodyFontFace: "Segoe UI",
  colors: {
    dk1: "142430", lt1: "FFFFFF", dk2: "0B2230", lt2: "EEF3F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "FAB219",
    accent5: "8FC1F2", accent6: "56656F", hlink: "2A78D6", folHlink: "1C5CAB",
  },
};
// hex for options that take hex only (backgrounds, shadows, chart colours, grid lines)
const HEX = {
  ink: "142430", navy: "0B2230", navy2: "12384D", light: "EEF3F6", water: "2A78D6", laterite: "EB6834",
  sky: "8FC1F2", muted: "56656F", grid: "E1E6EA", border: "D5DDE3", good: "0CA30C", warn: "FAB219",
  white: "FFFFFF", soft: "C9D6DE",
};
const HEAD = "+mj-lt"; // theme heading font, for big numbers and headers outside placeholders
const W = 13.333, M = 0.6, CW = W - 2 * M;
const FOOT = "SWOT Tank Water Tracker for Sri Lanka  ·  Group 6, JARVIS";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "SWOT Tank Water Tracker for Sri Lanka";
pres.subject = "IS4990 project proposal talk";
pres.author = "Group 6, JARVIS";
pres.company = "University of Moratuwa";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "COVER",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 7.3, h: 1.85, fontSize: 44,
      color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.95, w: 6.4, h: 0.95, fontSize: 18,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
for (const [name, bg, titleColor, footColor, numColor] of [
  ["LIGHT", HEX.white, C.text1, C.accent6, HEX.muted],
  ["DARK", HEX.navy, C.background1, C.accent5, HEX.sky],
]) {
  pres.defineSlideMaster({
    title: name,
    background: { color: bg },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.38, w: CW, h: 0.85, fontSize: 34,
        color: titleColor, align: "left", valign: "middle", margin: 0 }, text: "" } },
      { text: { text: FOOT, options: { x: M, y: 7.02, w: 8, h: 0.28, fontSize: 10, color: footColor, margin: 0 } } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.28, fontSize: 10, color: numColor, align: "right" },
  });
}

// ---------- helpers ----------
const iconCache = {};
async function icon(name, color = "#FFFFFF") {
  const key = name + color;
  if (!iconCache[key]) {
    const Comp = fa[name];
    if (!Comp) throw new Error("no icon " + name);
    const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { size: 256 })).replace(/currentColor/g, color);
    const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
    iconCache[key] = "image/png;base64," + buf.toString("base64");
  }
  return iconCache[key];
}
async function iconCircle(s, name, x, y, d, fill, color = "#FFFFFF") {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 },
    objectName: "icon " + name });
  const k = d * 0.5;
  s.addImage({ data: await icon(name, color), x: x + (d - k) / 2, y: y + (d - k) / 2, w: k, h: k, altText: "" });
}
function text(s, t, o) {
  s.addText(t, { isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 15, ...o });
}
const shadow = () => ({ type: "outer", color: HEX.navy, opacity: 0.13, blur: 9, offset: 2, angle: 90 });
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: o.fill || C.background2 },
    line: { color: o.line || o.fill || C.background2, width: o.lineWidth || 0.75, dashType: o.dash || "solid" },
    shadow: o.shadow ? shadow() : undefined, objectName: o.name || "card" });
}
function arrow(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.LINE, { x, y, w, h, flipV: !!o.up,
    line: { color: o.color || HEX.muted, width: 1.75, endArrowType: "triangle" }, objectName: "arrow" });
}
// place a picture inside a box, keeping its shape; align "left", "center" or "right" in the box
async function fit(s, file, x, y, wMax, hMax, align, altText) {
  const { width, height } = await sharp(file).metadata();
  let w = wMax, h = (wMax * height) / width;
  if (h > hMax) { h = hMax; w = (hMax * width) / height; }
  const dx = align === "right" ? wMax - w : align === "center" ? (wMax - w) / 2 : 0;
  s.addImage({ path: file, x: x + dx, y, w, h, altText });
  return { x: x + dx, y, w, h };
}
const fmt = (d) => new Date(d + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

async function build() {
  let s;

  // 1. Cover
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
  s.addImage({ path: path.join(CARDS, "cover.jpg"), x: 0, y: 0, w: W, h: 7.5,
    altText: "Sentinel-2 satellite view of the tanks around Mahakanadarawa" });
  text(s, "IS4990 COMPREHENSIVE GROUP PROJECT  ·  PROJECT PROPOSAL",
    { x: 0.7, y: 1.5, w: 7.6, h: 0.35, fontSize: 13, color: C.accent5, bold: true, charSpacing: 1.5 });
  s.addText("SWOT Tank Water Tracker for Sri Lanka", { placeholder: "title" });
  s.addText("An AI system that estimates how much water each tank holds, using free satellite data", { placeholder: "body" });
  text(s, "Group 6  ·  JARVIS", { x: 0.7, y: 5.2, w: 6.8, h: 0.4, fontSize: 17, color: C.background1, fontFace: HEAD });
  text(s, "Supervisors: Dr. Upeksha Ganegoda (IT)  ·  Prof. (Mrs) Thanuja Sandanayake (IDS)",
    { x: 0.7, y: 5.65, w: 7.6, h: 0.35, fontSize: 12.5, color: HEX.soft });
  text(s, "October 2026", { x: 0.7, y: 6.02, w: 3, h: 0.35, fontSize: 12.5, color: HEX.soft });
  text(s, `Tanks around Mahakanadarawa, Sentinel-2, ${fmt(pics.hero.date)}. Contains modified Copernicus Sentinel data 2026.`,
    { x: 6.4, y: 7.05, w: 6.33, h: 0.28, fontSize: 9, color: HEX.soft, align: "right" });

  // 2. Why it matters
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Introduction" });
  s.addText("Why: dry-zone farming depends on tank water", { placeholder: "title" });
  const whys = [
    ["~23,000", "small tanks of 80 ha or less in Sri Lanka, the base of dry-zone farming", "National Tanks Survey, in Lopane et al., 2025"],
    ["2", "seasons a year. Before each, farmers plan the crop by the water in the tank", "Cultivation meetings at village tanks, Oka et al., 2015"],
    ["1,160", "small tanks damaged by Cyclone Ditwah in November 2025", "Department of Agrarian Development, December 2025"],
  ];
  whys.forEach(([big, label, src], i) => {
    const y = 1.55 + i * 1.42;
    text(s, big, { x: M, y, w: 2.75, h: 0.85, fontSize: 40, color: i === 2 ? C.accent2 : C.accent1, fontFace: HEAD, valign: "middle" });
    text(s, label, { x: 3.5, y: y + 0.02, w: 4.2, h: 0.62, fontSize: 15 });
    text(s, src, { x: 3.5, y: y + 0.7, w: 4.2, h: 0.3, fontSize: 10.5, color: C.accent6 });
  });
  text(s, "Knowing each tank's water helps farmers plan the season, and helps officers share water and prepare for floods.",
    { x: M, y: 5.45, w: 6.85, h: 0.62, fontSize: 16, italic: true, color: C.text2 });
  text(s, "But today, nobody can say how much water most tanks hold.",
    { x: M, y: 6.22, w: 6.85, h: 0.45, fontSize: 16, bold: true, color: C.accent2 });
  s.addImage({ path: path.join(CARDS, "problem.jpg"), x: 7.93, y: 1.5, w: 4.8, h: 5.3,
    altText: "Satellite image of the study area with every mapped tank shown in blue" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.08, y: 5.92, w: 4.5, h: 0.74, rectRadius: 0.08,
    fill: { color: HEX.navy, transparency: 12 }, line: { color: HEX.navy, width: 0 }, objectName: "image caption" });
  text(s, [
    { text: `${pics.problem.tanks_6ha} tanks of 6.25 ha or more in our study area (blue)`, options: { fontSize: 11.5, bold: true, breakLine: true } },
    { text: "Sentinel-2, 28 Feb 2026 · outlines © OpenStreetMap contributors", options: { fontSize: 9 } },
  ], { x: 8.22, y: 6.0, w: 4.25, h: 0.6, color: C.background1, valign: "middle" });

  // 3. The problem we solve
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Introduction" });
  s.addText("The problem we solve", { placeholder: "title" });
  card(s, M, 1.4, CW, 1.25, { fill: HEX.navy, name: "problem statement" });
  text(s, "THE PROBLEM", { x: M + 0.35, y: 1.55, w: 4, h: 0.3, fontSize: 12, bold: true, color: C.accent5, charSpacing: 1.5 });
  text(s, "For most Sri Lankan tanks, nobody can say how much water they hold right now, and there is no low-cost way to measure it often.",
    { x: M + 0.35, y: 1.86, w: CW - 0.7, h: 0.7, fontSize: 21, fontFace: HEAD, color: C.background1 });
  text(s, "WHY IT IS STILL UNSOLVED", { x: M, y: 2.88, w: 6, h: 0.3, fontSize: 12, bold: true, color: C.accent6, charSpacing: 1.5 });
  const gaps = [
    ["FaGaugeHigh", "No gauges on village tanks", "Only 118 reservoirs publish daily water levels", "Irrigation Department, Reservoir Status, 25 Oct 2026"],
    ["FaMap", "Maps show area, not volume", "IFAD maps over 10,650 water bodies, but by area only", "IFAD, 2026"],
    ["FaCloud", "Clouds and water plants hide the water", "Only 17–22% of Sentinel-2 dates were clear over our tanks; plants cover 3 of our 8 village tanks", "Our data check, October 2026"],
    ["FaSatellite", "SWOT gives height, not volume", "Height only, and not yet checked on Sri Lankan tanks", "NASA and CNES satellite, launched December 2022"],
  ];
  const gw = (CW - 0.25) / 2;
  for (const [i, [ic, head, body, src]] of gaps.entries()) {
    const x = M + (i % 2) * (gw + 0.25), y = 3.2 + Math.floor(i / 2) * 1.35;
    card(s, x, y, gw, 1.25, { name: "gap " + head });
    await iconCircle(s, ic, x + 0.25, y + 0.33, 0.58, C.accent1);
    text(s, head, { x: x + 1.05, y: y + 0.13, w: gw - 1.3, h: 0.34, fontSize: 16, fontFace: HEAD, valign: "middle" });
    text(s, body, { x: x + 1.05, y: y + 0.5, w: gw - 1.3, h: 0.44, fontSize: 12.5 });
    text(s, src, { x: x + 1.05, y: y + 0.96, w: gw - 1.3, h: 0.2, fontSize: 9.5, color: C.accent6 });
  }
  card(s, M, 5.97, CW, 0.9, { fill: C.background1, line: HEX.border, shadow: true, name: "our answer" });
  await iconCircle(s, "FaDroplet", M + 0.3, 6.12, 0.6, C.accent2);
  text(s, [
    { text: "Our answer: ", options: { bold: true, color: C.accent2 } },
    { text: "an AI system that turns free satellite data into each tank's storage, in m³ and % full, every 5–12 days, checked against gauges.", options: { breakLine: true } },
    { text: "For farmer organisations, the Department of Agrarian Development (village tanks) and the Irrigation Department (reservoirs).", options: { fontSize: 11.5, color: C.accent6 } },
  ], { x: M + 1.15, y: 6.0, w: CW - 1.4, h: 0.84, fontSize: 15, valign: "middle" });

  // 4. What SWOT gives vs what we add
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Introduction" });
  s.addText("SWOT gives the height. We turn it into water", { placeholder: "title" });
  const cols = [
    { x: M, head: "What SWOT gives", icon: "FaSatellite", fill: C.accent1, items: [
      "The water height of each lake, about twice every 21 days",
      "Only for lakes in its prior lake list",
      "Mostly \"suspect\" readings on small village tanks",
      "No volume, and nothing between passes",
    ] },
    { x: 6.83, head: "What our project adds", icon: "FaDroplet", fill: C.accent2, items: [
      "Water area from Sentinel-1 and Sentinel-2 by AI, even under cloud and water plants",
      "A height–area–volume curve for every tank",
      "Storage in m³ and % full, every 5–12 days",
      "Accuracy checked against Irrigation Department gauges",
      "A dashboard for irrigation officers and farmers",
    ] },
  ];
  for (const c of cols) {
    card(s, c.x, 1.45, 5.9, 4.2, { fill: C.background1, line: HEX.border, shadow: true, name: c.head });
    await iconCircle(s, c.icon, c.x + 0.35, 1.78, 0.66, c.fill);
    text(s, c.head, { x: c.x + 1.2, y: 1.83, w: 4.5, h: 0.55, fontSize: 21, fontFace: HEAD, valign: "middle" });
    text(s, c.items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < c.items.length - 1 } })),
      { x: c.x + 0.4, y: 2.72, w: 5.2, h: 2.8, fontSize: 16, paraSpaceAfter: 8 });
  }
  await iconCircle(s, "FaArrowRight", 6.365, 3.25, 0.6, C.accent2);
  text(s, "SWOT is the ruler. We turn its readings into water in the tank.",
    { x: M, y: 5.95, w: CW, h: 0.55, fontSize: 20, italic: true, color: C.text2, valign: "middle" });

  // 5. Meet SWOT
  pres.addSection({ title: "The satellites" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "The satellites" });
  s.addText("SWOT: a satellite that measures water height", { placeholder: "title" });
  s.addImage({ path: path.join(CARDS, "swot_satellite.jpg"), x: M, y: 1.45, w: 6.9, h: 3.88,
    altText: "Illustration of the SWOT satellite above Earth" });
  text(s, "Illustration: NASA/JPL-Caltech", { x: M, y: 5.38, w: 6.9, h: 0.25, fontSize: 9, color: HEX.soft });
  const sat = [["Dec 2022", "launched by NASA and CNES (France)"], ["120 km", "wide strip measured on each pass"],
    ["21 days", "until the same pass comes back"], ["2 passes", "see our tanks: 273 and 092"]];
  sat.forEach(([big, label], i) => {
    const x = M + i * (1.58 + 0.193);
    card(s, x, 5.72, 1.58, 1.08, { fill: HEX.navy2, name: "fact " + big });
    text(s, big, { x: x + 0.15, y: 5.8, w: 1.3, h: 0.42, fontSize: 21, color: C.background1, fontFace: HEAD, valign: "middle" });
    text(s, label, { x: x + 0.15, y: 6.23, w: 1.32, h: 0.5, fontSize: 10, color: HEX.soft });
  });
  await fit(s, path.join(IMG, "swot_passes.png"), 7.9, 1.4, 4.83, 5.45, "center",
    "Map of Sri Lanka with the strips of the 7 SWOT passes; passes 273 and 092 cross our study area");

  // 6. One pass
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "The satellites" });
  s.addText("What SWOT sees: one pass, every lake at once", { placeholder: "title" });
  text(s, String(heights.measured), { x: M, y: 1.4, w: 3.0, h: 1.15, fontSize: 66, color: C.accent1, fontFace: HEAD, valign: "middle" });
  text(s, "lakes measured in a single pass", { x: M, y: 2.55, w: 7.2, h: 0.42, fontSize: 19, fontFace: HEAD });
  text(s, `SWOT pass 273 on ${fmt(heights.date)}, in our 44 × 55 km study area`, { x: M, y: 2.98, w: 7.4, h: 0.35, fontSize: 14, color: C.accent6 });
  const seen = [
    ["FaRulerVertical", "Each colour is the water height SWOT measured, in metres above sea level."],
    ["FaCircleCheck", `All 5 of our reservoirs were measured on this pass, for example Hurulu Wewa at ${heights.heights["Hurulu Wewa"].toFixed(2)} m.`],
    ["FaEyeSlash", "SWOT cannot measure the strip right under itself, shown hatched on the map."],
  ];
  for (const [i, [ic, t]] of seen.entries()) {
    const y = 3.7 + i * 0.86;
    await iconCircle(s, ic, M, y, 0.56, C.accent1);
    text(s, t, { x: 1.4, y: y + 0.03, w: 6.4, h: 0.62, fontSize: 15, valign: "middle" });
  }
  text(s, "Data: NASA/JPL and CNES, SWOT Lake Single-Pass product, version D, from NASA PO.DAAC.",
    { x: M, y: 6.5, w: 7.4, h: 0.3, fontSize: 10.5, color: C.accent6 });
  await fit(s, path.join(IMG, "swot_heights.png"), 8.3, 1.3, 4.43, 5.6, "right",
    "Map of every lake SWOT measured in the study area on 11 October 2026, coloured by water height");

  // 7. Clouds and radar
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "The satellites" });
  s.addText("Clouds hide the tanks. Radar sees through them", { placeholder: "title" });
  const views = [
    ["hurulu_full_w.jpg", "Sentinel-2, clear day", `Hurulu Wewa, ${fmt(pics.hurulu_full.date)} · optical, 10 m`],
    ["hurulu_cloud_w.jpg", "Sentinel-2, monsoon cloud", `Hurulu Wewa, ${fmt(pics.hurulu_cloud.date)} · the tank is hidden`],
    ["hurulu_radar_w.jpg", "Sentinel-1 radar", `Hurulu Wewa, ${fmt(pics.hurulu_radar.date)} · water shows dark`],
  ];
  views.forEach(([file, head, sub], i) => {
    const x = M + i * (3.85 + 0.29);
    s.addImage({ path: path.join(CARDS, file), x, y: 1.4, w: 3.85, h: 4.06, altText: head + ", " + sub });
    text(s, head, { x, y: 5.58, w: 3.85, h: 0.34, fontSize: 15, fontFace: HEAD });
    text(s, sub, { x, y: 5.93, w: 3.85, h: 0.3, fontSize: 12, color: C.accent6 });
  });
  await iconCircle(s, "FaCloud", M, 6.38, 0.48, C.accent6);
  text(s, "Only 17–22% of Sentinel-2 dates were clear over our test tanks, so radar is a core input.",
    { x: 1.25, y: 6.38, w: 7.5, h: 0.48, fontSize: 14, valign: "middle" });
  text(s, "Contains modified Copernicus Sentinel data 2025–2026.", { x: 8.9, y: 6.38, w: 3.83, h: 0.48, fontSize: 9.5, color: C.accent6, align: "right", valign: "middle" });

  // 8. Study area
  pres.addSection({ title: "Study and method" });
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Study and method" });
  s.addText("Study area: 5 reservoirs and 8 village tanks", { placeholder: "title" });
  await fit(s, path.join(IMG, "study_map.png"), M, 1.45, 6.3, 5.4, "left",
    "Study area map east of Anuradhapura with five reservoirs and eight numbered village tanks");
  const area = [
    ["FaWater", C.accent1, "5 major reservoirs", "Hurulu Wewa, Mahakanadarawa, Manankattiya, Nachchaduwa and Nuwara Wewa, all with daily gauge readings"],
    ["FaSeedling", C.accent2, "8 village tanks", "0.4–5 km from Mahakanadarawa: 2 open water, 3 mixed and 3 mostly covered by water plants"],
    ["FaSatellite", C.accent1, "Seen by SWOT passes 273 and 092", "All 13 tanks, about twice every 21 days, with 47–90 usable readings each since July 2023"],
    ["FaMapLocationDot", C.accent6, "Why here", "Dry-zone tank cascades (ellanga), flat land, and official gauges to check against"],
  ];
  for (const [i, [ic, fill, head, sub]] of area.entries()) {
    const y = 1.5 + i * 1.33;
    await iconCircle(s, ic, 7.25, y, 0.6, fill);
    text(s, head, { x: 8.05, y: y - 0.02, w: 4.68, h: 0.4, fontSize: 17, fontFace: HEAD });
    text(s, sub, { x: 8.05, y: y + 0.38, w: 4.68, h: 0.8, fontSize: 13, color: C.accent6 });
  }

  // 9. How it works
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Study and method" });
  s.addText("How it works", { placeholder: "title" });
  card(s, M, 1.5, 2.75, 2.75, { name: "Sentinel inputs" });
  text(s, "Sentinel-1 and Sentinel-2", { x: M + 0.2, y: 1.65, w: 2.4, h: 0.34, fontSize: 15, fontFace: HEAD });
  text(s, "Radar and optical images, every 5–12 days", { x: M + 0.2, y: 2.0, w: 2.4, h: 0.5, fontSize: 11.5, color: C.accent6 });
  s.addImage({ path: path.join(CARDS, "thumb_s2.jpg"), x: M + 0.2, y: 2.68, w: 1.08, h: 1.14, altText: "Sentinel-2 view of Hurulu Wewa" });
  s.addImage({ path: path.join(CARDS, "thumb_s1.jpg"), x: M + 1.47, y: 2.68, w: 1.08, h: 1.14, altText: "Sentinel-1 radar view of Hurulu Wewa" });
  arrow(s, 3.35, 2.875, 0.6, 0);
  card(s, 3.95, 1.5, 2.75, 2.75, { name: "AI water map" });
  text(s, "AI water map", { x: 4.15, y: 1.65, w: 2.4, h: 0.34, fontSize: 15, fontFace: HEAD });
  text(s, "Open water, water under plants, and land", { x: 4.15, y: 2.0, w: 2.4, h: 0.5, fontSize: 11.5, color: C.accent6 });
  s.addImage({ path: path.join(CARDS, "thumb_water.jpg"), x: 4.78, y: 2.68, w: 1.08, h: 1.14, altText: "Water map of Hurulu Wewa" });
  card(s, M, 4.75, 6.1, 1.45, { name: "SWOT heights" });
  await iconCircle(s, "FaSatellite", M + 0.3, 5.16, 0.62, C.accent1);
  text(s, "SWOT water heights", { x: M + 1.15, y: 5.02, w: 4.7, h: 0.36, fontSize: 15, fontFace: HEAD });
  text(s, "From NASA's Hydrocron service since July 2023, with poor readings removed", { x: M + 1.15, y: 5.4, w: 4.75, h: 0.6, fontSize: 11.5, color: C.accent6 });
  arrow(s, 6.7, 2.875, 0.6, 0.425);
  arrow(s, 6.7, 4.3, 0.6, 1.175, { up: true });
  card(s, 7.3, 2.55, 2.65, 2.2, { fill: HEX.navy, name: "Height-area-volume curve" });
  text(s, "Height–area–volume curve", { x: 7.5, y: 2.72, w: 2.3, h: 0.7, fontSize: 16, fontFace: HEAD, color: C.background1 });
  text(s, "One per tank: each SWOT height paired with the AI water area of the same day", { x: 7.5, y: 3.5, w: 2.3, h: 1.1, fontSize: 11.5, color: HEX.soft });
  arrow(s, 9.95, 3.65, 0.45, 0);
  card(s, 10.4, 2.55, 2.33, 2.2, { name: "Storage" });
  text(s, [{ text: "Storage every", options: { breakLine: true } }, { text: "5–12 days" }],
    { x: 10.58, y: 2.72, w: 2.0, h: 0.7, fontSize: 16, fontFace: HEAD });
  text(s, "Each new water map becomes a volume, shown on a dashboard", { x: 10.58, y: 3.5, w: 2.0, h: 1.1, fontSize: 11.5, color: C.accent6 });
  arrow(s, 11.565, 4.75, 0, 0.4);
  card(s, 7.3, 5.15, 5.43, 1.05, { fill: C.background1, line: HEX.border, name: "Validation" });
  await iconCircle(s, "FaCheck", 7.5, 5.39, 0.56, C.accent3);
  text(s, "Checked against Irrigation Department gauges and official storage figures",
    { x: 8.25, y: 5.3, w: 4.3, h: 0.75, fontSize: 13.5, valign: "middle" });

  // 10. Hurulu Wewa in three pictures
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Study and method" });
  s.addText("Our method in three pictures: Hurulu Wewa", { placeholder: "title" });
  const steps = [
    ["hurulu_low_n.jpg", "Low", pics.hurulu_low, `Sentinel-2 ${fmt(pics.hurulu_low.date)} · SWOT ${fmt(pics.hurulu_low.swot_date)}`],
    ["hurulu_mid_n.jpg", "Half full", pics.hurulu_mid, `Sentinel-2 and SWOT, ${fmt(pics.hurulu_mid.date)}`],
    ["hurulu_full_n.jpg", "Full", pics.hurulu_full, `Sentinel-2 ${fmt(pics.hurulu_full.date)} · SWOT ${fmt(pics.hurulu_full.swot_date)}`],
  ];
  steps.forEach(([file, label, m, dates], i) => {
    const x = M + i * (3.85 + 0.29);
    s.addImage({ path: path.join(CARDS, file), x, y: 1.4, w: 3.85, h: 4.06, altText: `Hurulu Wewa ${label.toLowerCase()}, Sentinel-2` });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: 1.55, w: 1.15, h: 0.36, rectRadius: 0.08,
      fill: { color: HEX.navy, transparency: 20 }, line: { color: HEX.navy, width: 0 }, objectName: "label " + label });
    text(s, label, { x: x + 0.15, y: 1.55, w: 1.15, h: 0.36, fontSize: 12, color: C.background1, align: "center", valign: "middle", bold: true });
    text(s, [
      { text: `${m.swot_wse.toFixed(2)} m`, options: { fontSize: 28, fontFace: HEAD, color: C.background1 } },
      { text: "   SWOT height", options: { fontSize: 12, color: HEX.sky } },
    ], { x, y: 5.55, w: 3.85, h: 0.52, valign: "middle" });
    text(s, `${m.water_km2.toFixed(1)} km² of water`, { x, y: 6.08, w: 3.85, h: 0.32, fontSize: 15, color: C.accent5 });
    text(s, dates, { x, y: 6.42, w: 3.85, h: 0.28, fontSize: 10.5, color: HEX.soft });
  });

  // 11. Three years of SWOT levels
  pres.addSection({ title: "Evidence" });
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
  s.addText("Three years of SWOT water levels", { placeholder: "title" });
  text(s, [
    { text: "Monthly average of best-quality readings, in metres above each reservoir's lowest month.", options: { breakLine: true } },
    { text: "The sharp rise at the end of 2025 is Cyclone Ditwah.", options: { bold: true, color: C.accent2 } },
  ], { x: M, y: 1.25, w: CW, h: 0.6, fontSize: 14, color: C.accent6 });
  // pad Jan-Jul 2023 (before SWOT's science orbit) so each year label sits on January
  const pad = 7;
  const labels = Array(pad).fill("").concat(levels.labels).map((l, k) => (k % 12 === 0 ? String(2023 + k / 12) : ""));
  levels.series.forEach((ser, i) => {
    const x = M + i * (2.3 + 0.158);
    const values = Array(pad).fill(null).concat(ser.values);
    s.addChart(pres.charts.LINE, [{ name: ser.name, labels, values }], {
      x, y: 1.9, w: 2.3, h: 3.95,
      showTitle: true, title: ser.name, titleFontSize: 13, titleColor: HEX.ink, titleFontFace: HEAD,
      showLegend: false, chartColors: [HEX.water], lineSize: 2, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 5.5, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0",
      valAxisLabelFontSize: 9, valAxisLabelColor: HEX.muted, valAxisLabelFontFace: "+mn-lt", valAxisLineShow: false,
      catAxisLabelFontSize: 9, catAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt",
      catAxisLabelFrequency: 12, catAxisMajorTickMark: "none", catAxisLineShow: true,
      valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
      showValAxisTitle: i === 0, valAxisTitle: "metres", valAxisTitleFontSize: 9, valAxisTitleColor: HEX.muted,
      displayBlanksAs: "span", objectName: "chart " + ser.name,
    });
    text(s, `${ser.best} best-quality readings`, { x, y: 5.95, w: 2.3, h: 0.3, fontSize: 12, fontFace: HEAD });
    text(s, `rose and fell up to ${ser.range_m.toFixed(1)} m`, { x, y: 6.25, w: 2.3, h: 0.3, fontSize: 11.5, color: C.accent6 });
  });
  text(s, "Source: NASA PO.DAAC Hydrocron, SWOT lake product version D, July 2023 to October 2026. Gaps are months without a best-quality reading.",
    { x: M, y: 6.62, w: CW, h: 0.3, fontSize: 10, color: C.accent6 });

  // 12. Feasibility check
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
  s.addText("We tested the data before proposing", { placeholder: "title" });
  text(s, "Five quick checks in October 2026, using the real SWOT, Sentinel-2 and gauge data.",
    { x: M, y: 1.25, w: CW, h: 0.4, fontSize: 14, color: C.accent6 });
  const checks = [
    ["1 · LAKE IDS", "86%", "of Irrigation Department tanks in our area have a SWOT lake ID (43 of 50); 58% of all mapped tanks", "RISK"],
    ["2 · HEIGHTS", "26,641", "SWOT readings on 400 lakes since July 2023; village tanks get mostly \"suspect\" readings", "RISK"],
    ["3 · GAUGE CHECK", "40–64 cm", "a steady gap between SWOT and the gauges at all 5 reservoirs; one fixed correction each", "PASS"],
    ["4 · WATER AREA", "17–22%", "of Sentinel-2 dates clear; water plants hide the water in 3 village tanks", "RISK"],
    ["5 · TRAINING DATA", "2", "labelled water datasets ready to use: Sen1Floods11 and S1S2‑Water", "PASS"],
  ];
  for (const [i, [label, big, body, status]] of checks.entries()) {
    const x = M + i * (2.3 + 0.158);
    card(s, x, 1.85, 2.3, 3.55, { name: "check " + label });
    text(s, label, { x: x + 0.2, y: 2.03, w: 1.95, h: 0.3, fontSize: 11, bold: true, color: C.accent6, charSpacing: 0.5 });
    text(s, big, { x: x + 0.2, y: 2.38, w: 1.95, h: 0.62, fontSize: 30, fontFace: HEAD, color: C.accent1, valign: "middle" });
    text(s, body, { x: x + 0.2, y: 3.08, w: 1.92, h: 1.6, fontSize: 12.5 });
    const good = status !== "RISK";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 4.82, w: 1.1, h: 0.38, rectRadius: 0.08,
      fill: { color: good ? HEX.good : HEX.warn }, line: { color: good ? HEX.good : HEX.warn, width: 0 }, objectName: "status " + status });
    s.addImage({ data: await icon(good ? "FaCircleCheck" : "FaTriangleExclamation", good ? "#FFFFFF" : "#142430"),
      x: x + 0.3, y: 4.9, w: 0.22, h: 0.22, altText: "" });
    text(s, status, { x: x + 0.56, y: 4.82, w: 0.7, h: 0.38, fontSize: 11.5, bold: true, valign: "middle",
      color: good ? C.background1 : C.text1 });
  }
  card(s, M, 5.62, CW, 0.98, { fill: HEX.navy, name: "verdict" });
  await iconCircle(s, "FaCheck", M + 0.25, 5.81, 0.6, HEX.good);
  text(s, [
    { text: "Verdict: GO, with changes.  ", options: { fontSize: 18, fontFace: HEAD, color: C.background1 } },
    { text: "Sentinel-1 radar and a water-plant class became core parts of the plan, and RQ2 now tests SWOT's \"suspect\" readings.", options: { fontSize: 13.5, color: HEX.soft } },
  ], { x: M + 1.1, y: 5.68, w: CW - 1.3, h: 0.86, valign: "middle" });

  // 13. Why these tanks
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
  s.addText("Why these tanks? We checked all of Sri Lanka", { placeholder: "title" });
  text(s, "6th of 187", { x: M, y: 1.35, w: 7.6, h: 0.95, fontSize: 54, fontFace: HEAD, color: C.accent1, valign: "middle" });
  text(s, "Hurulu Wewa's rank among Sri Lanka's big lakes (1 km² or more), by best-quality SWOT readings",
    { x: M, y: 2.35, w: 7.4, h: 0.62, fontSize: 15 });
  const why = [
    ["FaRankingStar", "All 5 of ours are in the island's top 22, and they are the top 5 in our study area"],
    ["FaLocationDot", "The island's top tanks are spread from Puttalam to Hambantota, far from our village tanks"],
    ["FaSatellite", "Ours sit next to the 8 village tanks and share the same 2 SWOT passes"],
    ["FaGaugeHigh", "All 5 have daily Irrigation Department gauges; 2 of the island's top 5 tanks do not"],
  ];
  for (const [i, [ic, t]] of why.entries()) {
    const y = 3.22 + i * 0.8;
    await iconCircle(s, ic, M, y, 0.52, C.accent1);
    text(s, t, { x: 1.35, y: y - 0.04, w: 6.6, h: 0.62, fontSize: 14.5, valign: "middle" });
  }
  text(s, "Best quality: quality flag 0, crossover flag 0–1, dark water 10% or less (Köhn and Nielsen, 2026).",
    { x: M, y: 6.5, w: 7.6, h: 0.3, fontSize: 10, color: C.accent6 });
  await fit(s, path.join(IMG, "ranking.png"), 8.5, 1.3, 4.23, 5.6, "right",
    "Map of Sri Lanka's 187 big lakes coloured by best-quality SWOT readings, with our five reservoirs circled");

  // 14. Research questions
  pres.addSection({ title: "Plan" });
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Plan" });
  s.addText("Research questions", { placeholder: "title" });
  const rqs = [
    ["How accurately can deep learning map tank water from Sentinel-1 and Sentinel-2, even under cloud or water plants?", "Measured by IoU and F1 against hand-labelled water masks"],
    ["How accurate is SWOT on reservoirs and village tanks, and can a learned model tell which readings to trust better than SWOT's own flags?", "Measured by error in cm against gauges, on reservoirs the model never saw"],
    ["Does AI water area plus SWOT height give storage close to official and measured values?", "Measured by error as a % of capacity"],
    ["Can a deep-learning model forecast each tank's storage 2–4 weeks ahead better than a \"no change\" forecast?", "Measured by error as a % of capacity at 1, 2 and 4 weeks"],
  ];
  for (const [i, [q, how]] of rqs.entries()) {
    const x = i % 2 ? 6.78 : M, y = i < 2 ? 1.5 : 4.0;
    const stretch = false; // RQ4 became core on 28 Oct, when each member got a model to train
    card(s, x, y, 5.95, 2.3, { fill: stretch ? C.background1 : C.background2, line: stretch ? HEX.border : undefined,
      dash: stretch ? "dash" : undefined, lineWidth: stretch ? 1.25 : undefined, name: "RQ" + (i + 1) });
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fill: { color: stretch ? C.accent6 : C.accent1 },
      line: { color: stretch ? C.accent6 : C.accent1, width: 0 }, objectName: "RQ number" });
    text(s, String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fontSize: 20, fontFace: HEAD, color: C.background1, align: "center", valign: "middle" });
    text(s, q, { x: x + 1.15, y: y + 0.28, w: 4.55, h: 1.35, fontSize: 16 });
    text(s, how, { x: x + 1.15, y: y + 1.72, w: 4.55, h: 0.36, fontSize: 11.5, color: C.accent6 });
  }

  // 15. Who does what
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Plan" });
  s.addText("Who does what: one model each", { placeholder: "title" });
  text(s, "Each of us owns a research question from start to finish, and trains one model for it.",
    { x: M, y: 1.25, w: CW, h: 0.4, fontSize: 14, color: C.accent6 });
  const team = [
    { who: "MEMBER A", role: "AI water maps", icon: "FaBrain", fill: C.accent1, rqs: ["RQ1"], items: [
      "Labelled water masks of Sri Lankan tanks",
      "A water-plant class for covered tanks",
      "Model: a Sentinel-1 and Sentinel-2 water map (U-Net)",
    ], out: "water maps for all tanks" },
    { who: "MEMBER B", role: "SWOT and gauges", icon: "FaSatellite", fill: C.accent3, rqs: ["RQ2"], items: [
      "SWOT heights from Hydrocron, checked against gauges",
      "Gauge data: daily records and history",
      "Model: a SWOT trust model (gradient boosting)",
    ], out: "accuracy report and trust model" },
    { who: "MEMBER C", role: "Storage and forecast", icon: "FaChartLine", fill: C.accent2, rqs: ["RQ3", "RQ4"], items: [
      "Height–area–volume curve for each tank",
      "Storage every 5–12 days, checked against official figures",
      "Model: a 2–4 week storage forecast (LSTM)",
    ], out: "storage, forecast and dashboard" },
  ];
  const tw = (CW - 2 * 0.25) / 3;
  for (const [i, m] of team.entries()) {
    const x = M + i * (tw + 0.25);
    card(s, x, 1.75, tw, 4.15, { fill: C.background1, line: HEX.border, shadow: true, name: m.who });
    await iconCircle(s, m.icon, x + 0.3, 1.95, 0.6, m.fill);
    text(s, m.who, { x: x + 1.05, y: 1.95, w: tw - 1.3, h: 0.6, fontSize: 12, bold: true, color: C.accent6, charSpacing: 1.5, valign: "middle" });
    text(s, m.role, { x: x + 0.3, y: 2.65, w: tw - 0.55, h: 0.45, fontSize: 19, fontFace: HEAD, valign: "middle" });
    let px = x + 0.3;
    for (const rq of m.rqs) {
      const stretch = rq.includes("stretch");
      const pw = stretch ? 1.3 : 0.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: 3.2, w: pw, h: 0.34, rectRadius: 0.08,
        fill: { color: stretch ? C.background1 : m.fill }, line: { color: stretch ? C.accent6 : m.fill, width: 1, dashType: stretch ? "dash" : "solid" },
        objectName: rq });
      text(s, rq, { x: px, y: 3.2, w: pw, h: 0.34, fontSize: 12, bold: true, align: "center", valign: "middle",
        color: stretch ? C.accent6 : C.background1 });
      px += pw + 0.12;
    }
    text(s, m.items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < m.items.length - 1 } })),
      { x: x + 0.3, y: 3.7, w: tw - 0.55, h: 1.6, fontSize: 14, paraSpaceAfter: 6 });
    s.addShape(pres.shapes.LINE, { x: x + 0.3, y: 5.36, w: tw - 0.6, h: 0, line: { color: HEX.border, width: 1 }, objectName: "divider" });
    text(s, [
      { text: "Output: ", options: { bold: true } },
      { text: m.out },
    ], { x: x + 0.3, y: 5.42, w: tw - 0.55, h: 0.4, fontSize: 12.5, valign: "middle" });
  }
  card(s, M, 6.05, CW, 0.75, { fill: HEX.navy, name: "shared work" });
  await iconCircle(s, "FaPeopleGroup", M + 0.25, 6.18, 0.5, C.accent5, "#0B2230");
  text(s, [
    { text: "All three: ", options: { bold: true, color: C.accent5 } },
    { text: "field visits on SWOT pass dates, labelling water masks, the thesis and the research paper." },
  ], { x: M + 0.95, y: 6.1, w: CW - 1.2, h: 0.65, fontSize: 15, color: C.background1, valign: "middle" });

  // 16-18. Each member's work: the data analysis and the technical work
  const members = [
    { who: "Member A", role: "AI water maps", rqs: ["RQ1"], fill: C.accent1, icon: "FaBrain",
      q: "How accurately can deep learning map tank water from Sentinel-1 and Sentinel-2, even under cloud or water plants?",
      analysis: [
        "Label water masks of our tanks, with a \"water under plants\" class",
        "Score the maps: IoU and F1 against the labels",
        "Area error against Kanhindawewa's drone survey, by sensor, season and plant cover",
        "Compare with the simple rules and with SWOT's own area on the same days",
      ],
      tech: [
        "U-Net (EfficientNet-B0) trained on S1S2-Water, then fine-tuned on our tanks",
        "Inputs: Sentinel-1 VV and VH, NISAR L-band radar, Sentinel-2, slope",
        "Pipeline: find each picture day, download it, map the water, measure the area",
      ],
      out: "water area for every tank on every picture day, with its accuracy" },
    { who: "Member B", role: "SWOT and gauges", rqs: ["RQ2"], fill: C.accent3, icon: "FaSatellite",
      q: "How accurate is SWOT on reservoirs and village tanks, and can a learned model tell which readings to trust better than SWOT's own flags?",
      analysis: [
        "SWOT minus gauge on every pass: the average gap and the spread in cm",
        "Labels for the model: each reading's error at the 117 department reservoirs",
        "The trust model against SWOT's own flags, on reservoirs it never saw",
        "How a height error turns into a volume error",
      ],
      tech: [
        "Hydrocron pipeline: each tank's lake ID, every pass",
        "Trust model: gradient boosting (XGBoost or LightGBM) on SWOT's quality fields, track distance, lake size and season",
        "Gauge zero in metres above sea level: full supply level minus full supply depth",
        "Daily logger for the department's sheet; storage history turned back into gauge heights",
      ],
      out: "SWOT accuracy report, the fixed gap for each reservoir, and a trust model that scores every reading" },
    { who: "Member C", role: "Storage and forecast", rqs: ["RQ3", "RQ4"], fill: C.accent2, icon: "FaChartLine",
      q: "Does AI water area plus SWOT height give storage close to official values, and can a deep-learning model forecast it 2–4 weeks ahead?",
      analysis: [
        "Each tank's area–height curve A(h) from its dots",
        "Storage change against the department's daily storage",
        "Test the method on Kanhindawewa's drone table, and against Imbulana's formula",
        "Forecast error at 1, 2 and 4 weeks, against \"no change\" and the seasonal average",
      ],
      tech: [
        "Storage engine: dots, curve, thin layers and safe limits, after every pass",
        "Forecast model: an LSTM with a gradient-boosting baseline, trained on 117 reservoirs' daily storage (2020–2026), rain and evaporation",
        "Dashboard: storage, safe limits and the forecast for each tank",
        "Automatic updates with GitHub Actions",
      ],
      out: "storage for every tank with safe limits, and a 2–4 week forecast, on a dashboard" },
  ];
  const mw = (CW - 0.3) / 2;
  for (const m of members) {
    s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Plan" });
    s.addText(`${m.who}: ${m.role}`, { placeholder: "title" });
    let px = M;
    for (const rq of m.rqs) {
      const stretch = rq.includes("stretch");
      const pw = stretch ? 1.3 : 0.7;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: 1.32, w: pw, h: 0.34, rectRadius: 0.08,
        fill: { color: stretch ? C.background1 : m.fill }, line: { color: stretch ? C.accent6 : m.fill, width: 1, dashType: stretch ? "dash" : "solid" },
        objectName: rq });
      text(s, rq, { x: px, y: 1.32, w: pw, h: 0.34, fontSize: 12, bold: true, align: "center", valign: "middle",
        color: stretch ? C.accent6 : C.background1 });
      px += pw + 0.12;
    }
    text(s, m.q, { x: px + 0.1, y: 1.27, w: M + CW - px - 0.1, h: 0.45, fontSize: 14, color: C.accent6, valign: "middle" });
    const cols = [["Data analysis", "FaChartColumn", m.analysis], ["Technical work", "FaGears", m.tech]];
    for (const [i, [head, ic, items]] of cols.entries()) {
      const x = M + i * (mw + 0.3);
      card(s, x, 1.95, mw, 3.95, i === 0 ? { name: head } : { fill: C.background1, line: HEX.border, shadow: true, name: head });
      await iconCircle(s, ic, x + 0.3, 2.13, 0.5, i === 0 ? m.fill : C.accent6);
      text(s, head, { x: x + 0.95, y: 2.13, w: mw - 1.2, h: 0.5, fontSize: 18, fontFace: HEAD, valign: "middle" });
      text(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
        { x: x + 0.3, y: 2.8, w: mw - 0.6, h: 3.0, fontSize: 15, paraSpaceAfter: 8 });
    }
    card(s, M, 6.05, CW, 0.75, { fill: HEX.navy, name: "output" });
    await iconCircle(s, "FaFlagCheckered", M + 0.25, 6.18, 0.5, C.accent5, "#0B2230");
    text(s, [
      { text: "Output: ", options: { bold: true, color: C.accent5 } },
      { text: m.out },
    ], { x: M + 0.95, y: 6.1, w: CW - 1.2, h: 0.65, fontSize: 15, color: C.background1, valign: "middle" });
    // the notes sit in this one statement, so strip_addnotes.py removes them for the public copies
  }

  // 19. Deliverables and timeline
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Plan" });
  s.addText("What we will deliver, and when", { placeholder: "title" });
  const outs = [
    ["FaBrain", "A labelled dataset and an AI water map that works under cloud and water plants"],
    ["FaSatellite", "A SWOT trust model that scores every reading"],
    ["FaChartLine", "Curves and storage every 5–12 days for all 13 tanks"],
    ["FaGaugeHigh", "A 2–4 week storage forecast, shown on a dashboard"],
    ["FaFileLines", "A research paper, for example for UoM's ICITR conference"],
  ];
  for (const [i, [ic, t]] of outs.entries()) {
    const y = 1.55 + i * 1.0;
    await iconCircle(s, ic, M, y, 0.6, i === 3 ? C.accent2 : C.accent1);
    text(s, t, { x: 1.45, y: y + 0.02, w: 5.2, h: 0.58, fontSize: 16, valign: "middle" });
  }
  card(s, 7.15, 1.4, 5.58, 5.4, { name: "timeline" });
  const miles = [
    ["Nov 2026", "Proposal, ethics and data requests"],
    ["15 Dec 2026", "Study area confirmed after the cascade and name checks"],
    ["Dec 2026 – Apr 2027", "Labelled data, gauge readings, and the three models: water map, trust model, forecast"],
    ["Mar – May 2027", "Storage estimates, validation and dashboard"],
    ["30 Jun 2027", "Thesis, paper and final submission"],
  ];
  s.addShape(pres.shapes.LINE, { x: 7.67, y: 1.85, w: 0, h: 4.3, line: { color: HEX.border, width: 2 }, objectName: "timeline line" });
  miles.forEach(([when, what], i) => {
    const y = 1.72 + i * 1.02;
    const key = i === 1 || i === 4;
    s.addShape(pres.shapes.OVAL, { x: 7.55, y: y + 0.08, w: 0.24, h: 0.24, fill: { color: key ? C.accent2 : C.accent1 },
      line: { color: HEX.white, width: 1.5 }, objectName: "milestone " + when });
    text(s, when, { x: 8.0, y, w: 4.5, h: 0.36, fontSize: 14.5, fontFace: HEAD });
    text(s, what, { x: 8.0, y: y + 0.36, w: 4.5, h: 0.55, fontSize: 12.5, color: C.accent6 });
  });

  // 20. Datasets and links (every link opened and checked on 28 Oct 2026)
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Plan" });
  s.addText("Datasets and links", { placeholder: "title" });
  text(s, "Every dataset is free and public. Click a name to open it.", { x: M, y: 1.25, w: CW, h: 0.4, fontSize: 14, color: C.accent6 });
  const linkCols = [
    [
      ["Water height (SWOT)", [
        ["SWOT Lake and River Explorer", "click any tank to see its readings", EXPLORER],
        ["NASA Earthdata: SWOT lake product, version D", "the files we use", "https://search.earthdata.nasa.gov/search/granules?p=C3233944983-POCLOUD"],
        ["Hydrocron", "the free service that gives each lake's heights", "https://podaac.github.io/hydrocron/"],
      ]],
      ["Water area (pictures)", [
        ["Copernicus Browser", "Sentinel-1 and -2 pictures of our tanks",
          "https://browser.dataspace.copernicus.eu/?themeId=DEFAULT-THEME&dateMode=SINGLE&zoom=14&lat=8.3868&lng=80.5471&datasetId=S2_L2A_CDAS&fromTime=2024-03-10T00%3A00%3A00.000Z&toTime=2024-03-10T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR"],
        ["Planetary Computer: Sentinel-1 radar", "terrain-corrected radar", "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc"],
        ["ASF: NISAR radar", "L-band radar", "https://asf.alaska.edu/nisar/"],
        ["JAXA ALOS-2 radar on AWS", "L-band radar", "https://registry.opendata.aws/jaxa-alos-palsar2-scansar/"],
        ["S1S2-Water", "AI training data, 65 areas in 29 countries", "https://doi.org/10.5281/zenodo.8314175"],
      ]],
    ],
    [
      ["Ground truth", [
        ["Irrigation Department: reservoir water level", "the daily sheet", "https://docs.google.com/spreadsheets/d/e/2PACX-1vTcSGhi9RESl7CMCl1TQnrKe07Gx5Q696YiSB9jneIHqIP9lifpqSErgI3D5k9KtQXSdW5JpycIIr5e/pubhtml"],
        ["Irrigation Department: reservoir status", "web page", "https://irrigation.gov.lk/web/index.php?option=com_content&view=article&id=84&Itemid=201&lang=en"],
        ["Irrigation Department: reservoir map layer", "full supply levels", "https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/ArcGIS/rest/services/Scheme_Data_2/FeatureServer/10"],
        ["Kanhindawewa drone survey", "area and capacity every 1 ft (PDF)", "https://www.irrigation.gov.lk/web/images/Scheme-Maps/Anuradhapura/HuruluWewa/Area_Capacity_Kanhindawewa_01_07_2026.pdf"],
        ["Mahaweli Authority: reservoir status reports", "daily levels", "https://mahaweli.gov.lk/wms.html"],
      ]],
      ["Weather (for the forecast)", [
        ["CHIRPS", "daily rainfall", "https://www.chc.ucsb.edu/data/chirps"],
        ["ERA5", "weather, for evaporation", "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"],
      ]],
      ["Our work", [
        ["Project website: swot.mehara.work", "every tab, with the team plan", "https://swot.mehara.work/"],
      ]],
    ],
  ];
  const lw = (CW - 0.4) / 2;
  for (const [ci, groups] of linkCols.entries()) {
    const x = M + ci * (lw + 0.4);
    let y = 1.85;
    for (const [head, items] of groups) {
      text(s, head, { x, y, w: lw, h: 0.36, fontSize: 15, fontFace: HEAD, color: C.text1, valign: "middle" });
      y += 0.42;
      for (const [name, note, url] of items) {
        text(s, [
          { text: name, options: { hyperlink: { url }, bold: true, color: C.accent1 } },
          { text: "  " + note, options: { color: C.accent6 } },
        ], { x: x + 0.15, y, w: lw - 0.15, h: 0.32, fontSize: 12.5, valign: "middle" });
        y += 0.34;
      }
      y += 0.16;
    }
  }

  // 21. Thank you
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Plan" });
  s.addImage({ path: path.join(CARDS, "closing.jpg"), x: 0, y: 0, w: W, h: 7.5, altText: "" });
  s.addText("Thank you", { placeholder: "title" });
  text(s, "Questions and feedback are welcome.", { x: M, y: 1.28, w: 8, h: 0.45, fontSize: 20, color: HEX.soft });
  text(s, "Data sources", { x: M, y: 2.3, w: 6, h: 0.4, fontSize: 17, fontFace: HEAD, color: C.accent5 });
  const srcs = [
    "SWOT: NASA/JPL and CNES, Lake Single-Pass product version D (PO.DAAC, Hydrocron)",
    "Sentinel-1 and Sentinel-2: ESA Copernicus (Earth Search and Microsoft Planetary Computer)",
    "Reservoir levels and outlines: Irrigation Department of Sri Lanka",
    "Tank outlines © OpenStreetMap contributors; Sri Lanka boundary: geoBoundaries",
    "SWOT illustration: NASA/JPL-Caltech",
  ];
  text(s, srcs.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < srcs.length - 1 } })),
    { x: M, y: 2.8, w: 8.2, h: 3.0, fontSize: 13.5, color: C.background1, paraSpaceAfter: 8 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.95, y: 2.3, w: 2.78, h: 2.78, rectRadius: 0.1, fill: { color: HEX.white },
    line: { color: HEX.white, width: 0 }, objectName: "QR frame" });
  s.addImage({ path: path.join(IMG, "qr_explorer.png"), x: 10.1, y: 2.45, w: 2.48, h: 2.48,
    altText: "QR code to the SWOT Lake and River Explorer website" });
  text(s, [
    { text: "See SWOT readings for any lake:", options: { breakLine: true } },
    { text: "SWOT Lake and River Explorer", options: { hyperlink: { url: EXPLORER }, color: C.accent5, bold: true } },
  ], { x: 9.95, y: 5.25, w: 2.78, h: 0.8, fontSize: 12, color: HEX.soft, align: "center" });

  await pres.writeFile({ fileName: OUT });
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("saved", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
