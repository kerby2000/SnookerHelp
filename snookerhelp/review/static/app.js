const UI_VERSION = "v1.5.6";

const state = {
  reports: [],
  stem: null,
  tableState: null,
  review: null,
  assetsBase: "",
  selectedId: null,
  cropBackground: "source",
  primaryEvidenceKey: "source",
  overlaySelections: null,
  displaySettingsByBackground: {},
  referenceOverlays: {
    finalCenter: true,
    roughCenter: false,
    neighborEllipses: false,
  },
  sourceViewport: {
    scale: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    lastX: 0,
    lastY: 0,
    clickSuppressUntil: 0,
  },
};

const OVERLAY_COLUMNS = [
  ["points", "white dots", "accepted boundary pixels"],
  ["rejected", "red dots", "rejected boundary outliers"],
  ["ellipse", "dashed outline", "observed ellipse"],
  ["center", "center cross", "center of that row's fit"],
];

const MAP_ASSET_ORDER = [
  "gray_edge",
  "lab_delta_e",
  "chroma_difference",
  "ball_vs_cloth_probability",
  "physical_projection_band",
  "combined_boundary_score",
];

const $ = (id) => document.getElementById(id);

init().catch((error) => {
  document.body.innerHTML = `<pre class="fatal">${escapeHtml(String(error.stack || error))}</pre>`;
});

async function init() {
  $("uiVersion").textContent = UI_VERSION;
  const reportsPayload = await fetchJson("/api/reports");
  state.reports = reportsPayload.reports || [];
  $("reportSelect").innerHTML = state.reports
    .map((report) => `<option value="${escapeAttr(report.stem)}">${escapeHtml(report.stem)}</option>`)
    .join("");
  $("reportSelect").addEventListener("change", () => loadReport($("reportSelect").value));
  $("prevBall").addEventListener("click", () => selectByOffset(-1));
  $("nextBall").addEventListener("click", () => selectByOffset(1));
  $("sourceZoomOut").addEventListener("click", () => zoomSourceAtCenter(1 / 1.25));
  $("sourceZoomIn").addEventListener("click", () => zoomSourceAtCenter(1.25));
  $("sourceResetView").addEventListener("click", () => resetSourceViewport());
  $("sourceFitSelected").addEventListener("click", () => fitSourceToSelected());
  $("sourcePrintClusterOrder").addEventListener("click", () => printClusterOrder());
  $("legendOpen").addEventListener("click", () => showLegend());
  $("legendClose").addEventListener("click", () => hideLegend());
  setupSourceViewport();
  setupLegendDrag();
  if (state.reports.length) {
    const params = new URLSearchParams(location.search);
    await loadReport(
      params.get("report") || state.reports[0].stem,
      Number(params.get("ball")) || null,
    );
  }
}

async function loadReport(stem, selectedBallId = null) {
  state.stem = stem;
  $("reportSelect").value = stem;
  const payload = await fetchJson(`/api/table-state/${encodeURIComponent(stem)}`);
  state.tableState = payload.table_state;
  state.review = payload.review_feedback;
  state.assetsBase = payload.assets_base;
  state.overlaySelections = null;
  state.primaryEvidenceKey = "source";
  state.cropBackground = "source";
  resetSourceViewport({ renderNow: false });
  const firstBall = balls()[0];
  const requestedBall = balls().find((ball) => ball.ball_id === selectedBallId);
  state.selectedId = requestedBall ? requestedBall.ball_id : firstBall ? firstBall.ball_id : null;
  syncUrl();
  render();
}

function render() {
  renderSource();
  renderSelected();
  renderBallStats();
}

function balls() {
  return state.tableState?.balls || [];
}

function selectedBall() {
  return balls().find((ball) => ball.ball_id === state.selectedId) || null;
}

function renderSource() {
  const table = state.tableState;
  if (!table) return;
  $("imageSummary").textContent = `${table.image_name} · ${balls().length} estimates`;
  $("sourceImage").src = assetUrl(table.source_image_uri);
  const svg = $("sourceOverlay");
  clear(svg);
  const size = table.source_size_px || { width: 6000, height: 4000 };
  svg.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
  if ((table.table_corners_px || []).length === 4) {
    svg.appendChild(el("polygon", {
      points: table.table_corners_px.map((p) => p.join(",")).join(" "),
      class: "table-boundary",
    }));
  }
  for (const ball of balls()) {
    const p = ball.source_px;
    if (!p) continue;
    const cluster = clusterInfo(ball);
    const roleClass = cluster.cluster_role ? ` cluster-${cluster.cluster_role}` : "";
    const clusterTag = clusterDisplayTag(cluster);
    const group = el("g", {"data-ball": ball.ball_id});
    group.appendChild(el("circle", {
      cx: p[0], cy: p[1], r: Math.max(18, ball.radius_px || 38),
      class: `ball-marker${roleClass}${ball.ball_id === state.selectedId ? " selected" : ""}`,
    }));
    group.appendChild(labelAt(p[0] + 28, p[1] - 28, clusterTag ? `${ball.ball_id} ${clusterTag}` : `${ball.ball_id}`));
    group.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
    });
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selectedId = ball.ball_id;
      syncUrl();
      render();
    });
    svg.appendChild(group);
  }
  applySourceViewport();
}

function setupSourceViewport() {
  const stage = $("sourceStage");
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.0012);
    zoomSourceAtClient(factor, event.clientX, event.clientY);
  }, { passive: false });
  stage.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    state.sourceViewport.dragging = true;
    state.sourceViewport.lastX = event.clientX;
    state.sourceViewport.lastY = event.clientY;
    stage.classList.add("panning");
    stage.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });
  stage.addEventListener("pointermove", (event) => {
    const view = state.sourceViewport;
    if (!view.dragging) return;
    const dx = event.clientX - view.lastX;
    const dy = event.clientY - view.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 1) {
      view.clickSuppressUntil = Date.now() + 150;
    }
    view.panX += dx;
    view.panY += dy;
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    clampSourcePan();
    applySourceViewport();
  });
  const stopDrag = (event) => {
    state.sourceViewport.dragging = false;
    stage.classList.remove("panning");
    stage.releasePointerCapture?.(event.pointerId);
  };
  stage.addEventListener("pointerup", stopDrag);
  stage.addEventListener("pointercancel", stopDrag);
  stage.addEventListener("dblclick", () => fitSourceToSelected());
  window.addEventListener("resize", () => {
    clampSourcePan();
    applySourceViewport();
  });
}

function zoomSourceAtCenter(factor) {
  const rect = $("sourceStage").getBoundingClientRect();
  zoomSourceAtClient(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
}

function zoomSourceAtClient(factor, clientX, clientY) {
  const view = state.sourceViewport;
  const rect = $("sourceStage").getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const stageX = clientX - rect.left;
  const stageY = clientY - rect.top;
  const oldScale = view.scale;
  const newScale = clamp(oldScale * factor, 1, 12);
  const anchorX = (stageX - view.panX) / oldScale;
  const anchorY = (stageY - view.panY) / oldScale;
  view.scale = newScale;
  view.panX = stageX - anchorX * newScale;
  view.panY = stageY - anchorY * newScale;
  clampSourcePan();
  applySourceViewport();
}

function resetSourceViewport(options = {}) {
  state.sourceViewport.scale = 1;
  state.sourceViewport.panX = 0;
  state.sourceViewport.panY = 0;
  state.sourceViewport.dragging = false;
  if (options.renderNow !== false) applySourceViewport();
}

function fitSourceToSelected() {
  const ball = selectedBall();
  if (!ball?.source_px) return;
  const stagePoint = sourcePixelToStagePoint(ball.source_px);
  if (!stagePoint) return;
  const rect = $("sourceStage").getBoundingClientRect();
  const view = state.sourceViewport;
  view.scale = 5;
  view.panX = rect.width / 2 - stagePoint[0] * view.scale;
  view.panY = rect.height / 2 - stagePoint[1] * view.scale;
  clampSourcePan();
  applySourceViewport();
}

function sourcePixelToStagePoint(point) {
  const table = state.tableState;
  const size = table?.source_size_px;
  const rect = $("sourceStage").getBoundingClientRect();
  if (!size?.width || !size?.height || !rect.width || !rect.height) return null;
  const baseScale = Math.min(rect.width / size.width, rect.height / size.height);
  const offsetX = (rect.width - size.width * baseScale) / 2;
  const offsetY = (rect.height - size.height * baseScale) / 2;
  return [
    offsetX + Number(point[0]) * baseScale,
    offsetY + Number(point[1]) * baseScale,
  ];
}

function clampSourcePan() {
  const view = state.sourceViewport;
  const rect = $("sourceStage").getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const padX = Math.min(120, rect.width * 0.32);
  const padY = Math.min(120, rect.height * 0.32);
  view.panX = clamp(view.panX, rect.width - rect.width * view.scale - padX, padX);
  view.panY = clamp(view.panY, rect.height - rect.height * view.scale - padY, padY);
}

function applySourceViewport() {
  const view = state.sourceViewport;
  const transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`;
  $("sourceImage").style.transform = transform;
  $("sourceOverlay").style.transform = transform;
  $("sourceZoomLabel").textContent = `${Math.round(view.scale * 100)}%`;
  updateSourceLabelScale();
}

function updateSourceLabelScale() {
  const labelScale = sourceLabelScale();
  document.querySelectorAll("#sourceOverlay .source-label").forEach((label) => {
    const x = Number(label.getAttribute("data-x"));
    const y = Number(label.getAttribute("data-y"));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    label.setAttribute("transform", `translate(${x} ${y}) scale(${labelScale})`);
  });
}

function sourceLabelScale() {
  const table = state.tableState;
  const size = table?.source_size_px;
  const rect = $("sourceStage").getBoundingClientRect();
  if (!size?.width || !size?.height || !rect.width || !rect.height) return 1;
  const baseScale = Math.min(rect.width / size.width, rect.height / size.height);
  const zoomScale = Math.max(1, Number(state.sourceViewport?.scale || 1));
  const targetScreenFontPx = 13;
  const labelSvgFontPx = 28;
  return clamp(targetScreenFontPx / (labelSvgFontPx * baseScale * zoomScale), 0.12, 8);
}

function renderSelected() {
  const ball = selectedBall();
  if (!ball) {
    $("ballTitle").textContent = "Final estimate — no detected ball selected";
    $("ballPosition").textContent = `0 / ${balls().length}`;
    clear($("cropOverlay"));
    $("layerControls").innerHTML = "";
    renderDefinitionList("imageEvidence", [["Status", "No detected ball selected"]]);
    renderDefinitionList("physicalModel", [["Status", "No detected ball selected"]]);
    renderDefinitionList("confidencePanel", [["Status", "No detected ball selected"]]);
    $("mapControls").innerHTML = "";
    return;
  }
  const evidence = ball.evidence || {};
  $("ballTitle").textContent = `Final estimate — #${ball.ball_id} ${ball.label}`;
  const index = balls().findIndex((item) => item.ball_id === ball.ball_id);
  $("ballPosition").textContent = `${index + 1} / ${balls().length}`;
  renderCropOverlay(ball);
  renderLayerControls(ball);
  renderDefinitionList("imageEvidence", imageEvidenceRows(ball));
  renderDefinitionList("physicalModel", physicalModelRows(ball));
  renderDefinitionList("confidencePanel", confidenceRows(ball));
}

function renderCropOverlay(ball) {
  const svg = $("cropOverlay");
  clear(svg);
  if (!ball) return;
  const bounds = ball.evidence?.crop_bounds_px;
  if (!bounds) return;
  const [x0, y0, x1, y1] = bounds;
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const viewBox = cropViewBox(ball, x0, y0, w, h);
  svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const rasterUri = cropRasterUri(ball);
  if (rasterUri) {
    svg.appendChild(el("image", {
      href: assetUrl(rasterUri),
      x: 0,
      y: 0,
      width: w,
      height: h,
      preserveAspectRatio: "none",
      class: "crop-raster",
      style: cropRasterFilter(),
    }));
  }
  ensureOverlaySelections(ball);
  const rough = ball.evidence?.rough_center_px;
  const source = ball.source_px;
  for (const row of evidenceRows(ball)) {
    const selection = overlaySelectionFor(row.key);
    if (selection.points) drawBoundaryPoints(svg, ball, row.key, x0, y0);
    if (selection.rejected) drawRejectedBoundaryPoints(svg, ball, row.key, x0, y0);
    if (selection.ellipse) drawImageModel(svg, ball, row.key, x0, y0);
    if (selection.center) drawFitCenter(svg, ball, row.key, x0, y0);
  }
  if (state.referenceOverlays.neighborEllipses) drawNeighborEllipses(svg, ball, x0, y0);
  if (state.referenceOverlays.roughCenter && rough) drawCross(svg, rough[0] - x0, rough[1] - y0, "rough-cross", 16);
  if (state.referenceOverlays.finalCenter && source) drawCross(svg, source[0] - x0, source[1] - y0, "center-cross", 16);
}

function cropViewBox(ball, x0, y0, cropWidth, cropHeight) {
  const localPoints = [];
  const addPoint = (point) => {
    if (Array.isArray(point) && point.length >= 2) {
      const x = Number(point[0]);
      const y = Number(point[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        localPoints.push([x - x0, y - y0]);
      }
    }
  };
  addPoint(ball.source_px);
  addPoint(ball.evidence?.rough_center_px);
  const selectedImage = selectedImageModel(ball);
  addPoint(selectedImage?.center_px);
  ensureOverlaySelections(ball);
  for (const row of evidenceRows(ball)) {
    const selection = overlaySelectionFor(row.key);
    const variant = evidenceVariant(ball, row.key);
    if (selection.points) {
      for (const point of variant.points_px || []) addPoint(point);
    }
    if (selection.rejected) {
      for (const point of variant.rejected_points_px || []) addPoint(point);
    }
    if ((selection.ellipse || selection.center) && variant.ellipse_fit?.center_px) {
      addPoint(variant.ellipse_fit.center_px);
      const model = variant.ellipse_fit;
      if (model.major_axis_px && model.minor_axis_px) {
        const cx = Number(model.center_px[0]) - x0;
        const cy = Number(model.center_px[1]) - y0;
        const r = Math.max(Number(model.major_axis_px), Number(model.minor_axis_px)) / 2;
        localPoints.push([cx - r, cy - r], [cx + r, cy + r]);
      }
    }
  }
  if (state.referenceOverlays.neighborEllipses) {
    for (const model of neighborEllipses(ball)) {
      if (!model.center_px || !model.major_axis_px || !model.minor_axis_px) continue;
      const cx = Number(model.center_px[0]) - x0;
      const cy = Number(model.center_px[1]) - y0;
      const r = Math.max(Number(model.major_axis_px), Number(model.minor_axis_px)) / 2;
      if (Number.isFinite(cx) && Number.isFinite(cy) && Number.isFinite(r)) {
        localPoints.push([cx - r, cy - r], [cx + r, cy + r]);
      }
    }
  }
  const image = selectedImage || {};
  if (image.center_px && image.major_axis_px && image.minor_axis_px) {
    const cx = Number(image.center_px[0]) - x0;
    const cy = Number(image.center_px[1]) - y0;
    const r = Math.max(Number(image.major_axis_px), Number(image.minor_axis_px)) / 2;
    localPoints.push([cx - r, cy - r], [cx + r, cy + r]);
  }
  if (!localPoints.length) localPoints.push([cropWidth / 2, cropHeight / 2]);

  const xs = localPoints.map((point) => point[0]);
  const ys = localPoints.map((point) => point[1]);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(cropWidth, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(cropHeight, Math.max(...ys));
  const radiusPx = Number(ball.radius_px || 42);
  const pad = Math.max(8, radiusPx * 0.22);
  let width = Math.max(maxX - minX + 2 * pad, radiusPx * 3.45, 118);
  let height = Math.max(maxY - minY + 2 * pad, radiusPx * 3.0, 102);
  const stage = $("cropOverlay").getBoundingClientRect();
  const aspect = stage.width > 0 && stage.height > 0 ? stage.width / stage.height : 1.15;
  if (width / height < aspect) width = height * aspect;
  else height = width / aspect;
  width = Math.min(width, cropWidth);
  height = Math.min(height, cropHeight);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const x = clamp(centerX - width / 2, 0, cropWidth - width);
  const y = clamp(centerY - height / 2, 0, cropHeight - height);
  return { x, y, w: width, h: height };
}

function drawImageModel(svg, ball, evidenceKey, x0, y0) {
  const model = evidenceVariant(ball, evidenceKey).ellipse_fit;
  if (!model?.center_px) return;
  svg.appendChild(el("ellipse", {
    cx: model.center_px[0] - x0,
    cy: model.center_px[1] - y0,
    rx: (model.major_axis_px || 0) / 2,
    ry: (model.minor_axis_px || 0) / 2,
    transform: `rotate(${model.angle_deg || 0} ${model.center_px[0] - x0} ${model.center_px[1] - y0})`,
    class: "image-ellipse",
  }));
}

function drawBoundaryPoints(svg, ball, evidenceKey, x0, y0) {
  const points = evidenceVariant(ball, evidenceKey).points_px || [];
  for (const point of points) {
    svg.appendChild(el("circle", {
      cx: point[0] - x0,
      cy: point[1] - y0,
      r: 1.05,
      class: "boundary-dot",
    }));
  }
}

function drawRejectedBoundaryPoints(svg, ball, evidenceKey, x0, y0) {
  const records = rejectedPointRecords(evidenceVariant(ball, evidenceKey));
  for (const record of records) {
    const point = record.point_px;
    const dot = el("circle", {
      cx: point[0] - x0,
      cy: point[1] - y0,
      r: 1.35,
      class: "rejected-boundary-dot",
    });
    dot.appendChild(el("title", {}, rejectedReasonTitle(record)));
    svg.appendChild(dot);
  }
}

function rejectedPointRecords(variant) {
  const records = variant?.filter?.rejected_point_reasons || [];
  if (records.length) {
    return records
      .map((record) => ({
        point_px: Array.isArray(record.point_px) ? record.point_px : null,
        primary_reason: record.primary_reason || "unknown_rejected",
        reasons: record.reasons || [record.primary_reason || "unknown_rejected"],
      }))
      .filter((record) => record.point_px && record.point_px.length >= 2);
  }
  return (variant?.rejected_points_px || []).map((point) => ({
    point_px: point,
    primary_reason: "unknown_rejected",
    reasons: ["unknown_rejected"],
  }));
}

function drawNeighborEllipses(svg, ball, x0, y0) {
  for (const model of neighborEllipses(ball)) {
    if (!model?.center_px || !model.major_axis_px || !model.minor_axis_px) continue;
    const cx = Number(model.center_px[0]) - x0;
    const cy = Number(model.center_px[1]) - y0;
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
    const node = el("ellipse", {
      cx,
      cy,
      rx: Number(model.major_axis_px) / 2,
      ry: Number(model.minor_axis_px) / 2,
      transform: `rotate(${model.angle_deg || 0} ${cx} ${cy})`,
      class: "neighbor-ellipse",
    });
    const label = [
      model.id == null ? null : `#${model.id}`,
      model.label || null,
      model.distance_px == null ? null : `${fmt(model.distance_px)} px`,
    ].filter(Boolean).join(" ");
    if (label) node.appendChild(el("title", {}, `Neighbor ${label}`));
    svg.appendChild(node);

    if (model.id != null) {
      svg.appendChild(el("text", {
        x: cx,
        y: cy,
        class: "neighbor-label",
      }, `#${model.id}${model.label ? ` ${model.label}` : ""}`));
    }
  }
}

function neighborEllipses(ball) {
  return ball?.evidence?.diagnostics?.neighbor_ellipses || [];
}

function rejectedReasonTitle(record) {
  return `Rejected: ${(record.reasons || [record.primary_reason]).map(displayLabel).join(", ")}`;
}

function drawFitCenter(svg, ball, evidenceKey, x0, y0) {
  const center = evidenceVariant(ball, evidenceKey).ellipse_fit?.center_px;
  if (!center) return;
  drawCross(svg, center[0] - x0, center[1] - y0, "fit-center-cross", 13);
}

function drawScaleBar(svg, ball, viewBox) {
  const radiusPx = Number(ball.radius_px || 0);
  const radiusMm = Number(ball.radius_mm || 26.25);
  if (!radiusPx || !radiusMm) return;
  const scaleMm = 10;
  const scalePx = Math.max(8, radiusPx / radiusMm * scaleMm);
  const x = viewBox.x + viewBox.w * 0.055;
  const y = viewBox.y + viewBox.h * 0.93;
  const tick = Math.max(3, viewBox.h * 0.018);
  svg.appendChild(el("line", { x1: x, y1: y, x2: x + scalePx, y2: y, class: "scale-line" }));
  svg.appendChild(el("line", { x1: x, y1: y - tick, x2: x, y2: y + tick, class: "scale-line" }));
  svg.appendChild(el("line", { x1: x + scalePx, y1: y - tick, x2: x + scalePx, y2: y + tick, class: "scale-line" }));
  svg.appendChild(el("text", { x, y: y - tick * 1.9, class: "scale-text" }, `${scaleMm} mm`));
}

function renderLayerControls(ball) {
  ensureOverlaySelections(ball);
  renderMapControls(ball);
  renderOverlayMatrix(ball);
}

function renderMapControls(ball) {
  const rows = evidenceRows(ball).filter((row) => row.hasBackground);
  const hasSelected = rows.some((row) => row.key === state.cropBackground);
  if (!hasSelected) state.cropBackground = "source";
  const selected = evidenceRows(ball).find((row) => row.key === state.cropBackground);
  const display = displaySettingsFor(state.cropBackground);
  $("mapControls").innerHTML = `
    <div class="active-background">
      <span>Evidence background</span>
      <strong>${escapeHtml(selected?.label || "Source image")}</strong>
    </div>
    <p class="map-description">${escapeHtml(selected?.description || "")}</p>
    <div class="display-tuning">
      <div class="display-tuning-header">
        <strong>Display tuning</strong>
        <span class="muted">view only; fit is unchanged</span>
      </div>
      <label>
        Brightness
        <input type="range" min="40" max="180" step="5" value="${escapeAttr(display.brightness)}" data-display-control="brightness">
        <span>${escapeHtml(display.brightness)}%</span>
      </label>
      <label>
        Contrast
        <input type="range" min="40" max="220" step="5" value="${escapeAttr(display.contrast)}" data-display-control="contrast">
        <span>${escapeHtml(display.contrast)}%</span>
      </label>
      <label class="display-checkbox">
        <input type="checkbox" ${display.invert ? "checked" : ""} data-display-control="invert">
        Invert background
      </label>
      <button type="button" class="small-button" data-display-reset>Reset display</button>
    </div>
  `;
  $("mapControls").querySelectorAll("[data-display-control]").forEach((input) => {
    input.addEventListener("input", () => {
      const settings = displaySettingsFor(state.cropBackground);
      if (input.dataset.displayControl === "invert") {
        settings.invert = input.checked;
      } else {
        settings[input.dataset.displayControl] = Number(input.value);
      }
      renderCropOverlay(selectedBall());
      renderMapControls(selectedBall());
    });
  });
  $("mapControls").querySelector("[data-display-reset]")?.addEventListener("click", () => {
    state.displaySettingsByBackground[state.cropBackground] = defaultDisplaySettings();
    renderCropOverlay(selectedBall());
    renderMapControls(selectedBall());
  });
}

function defaultDisplaySettings() {
  return { brightness: 100, contrast: 100, invert: false };
}

function displaySettingsFor(backgroundKey) {
  const key = backgroundKey || "source";
  if (!state.displaySettingsByBackground[key]) {
    state.displaySettingsByBackground[key] = defaultDisplaySettings();
  }
  return state.displaySettingsByBackground[key];
}

function cropRasterFilter() {
  const display = displaySettingsFor(state.cropBackground);
  const filters = [
    `brightness(${Number(display.brightness || 100)}%)`,
    `contrast(${Number(display.contrast || 100)}%)`,
  ];
  if (display.invert) filters.push("invert(1)");
  return `filter: ${filters.join(" ")};`;
}

function renderOverlayMatrix(ball) {
  const rows = evidenceRows(ball);
  $("layerControls").innerHTML = `
    <table class="overlay-matrix">
      <thead>
        <tr>
          <th>Evidence view</th>
          <th title="Diagnostic view score, not ground truth accuracy">Score</th>
          ${OVERLAY_COLUMNS.map(([key, label]) => `
            <th title="${escapeAttr(label)}">
              <span class="matrix-pictogram matrix-${escapeAttr(key)}"></span>
              <span>${escapeHtml(label)}</span>
            </th>
          `).join("")}
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => overlayMatrixRow(ball, row)).join("")}
      </tbody>
    </table>
    <div class="reference-overlays">
      <label><input type="checkbox" data-reference-overlay="finalCenter" ${state.referenceOverlays.finalCenter ? "checked" : ""}> final source center</label>
      <label><input type="checkbox" data-reference-overlay="roughCenter" ${state.referenceOverlays.roughCenter ? "checked" : ""}> rough detector center</label>
      <label><input type="checkbox" data-reference-overlay="neighborEllipses" ${state.referenceOverlays.neighborEllipses ? "checked" : ""}> neighbor ellipses</label>
    </div>
  `;
  $("layerControls").querySelectorAll("button[data-evidence-row]").forEach((button) => {
    button.addEventListener("click", () => {
      activateEvidenceRow(ball, button.dataset.evidenceRow);
      renderCropOverlay(selectedBall());
      renderLayerControls(selectedBall());
      renderDefinitionList("imageEvidence", imageEvidenceRows(selectedBall()));
      renderDefinitionList("physicalModel", physicalModelRows(selectedBall()));
      renderBallStats();
    });
  });
  $("layerControls").querySelectorAll("input[data-overlay-row]").forEach((input) => {
    input.addEventListener("change", () => {
      const rowKey = input.dataset.overlayRow;
      const columnKey = input.dataset.overlayColumn;
      overlaySelectionFor(rowKey)[columnKey] = input.checked;
      state.primaryEvidenceKey = rowKey;
      renderCropOverlay(selectedBall());
      renderLayerControls(selectedBall());
      renderDefinitionList("imageEvidence", imageEvidenceRows(selectedBall()));
      renderDefinitionList("physicalModel", physicalModelRows(selectedBall()));
      renderBallStats();
    });
  });
  $("layerControls").querySelectorAll("input[data-reference-overlay]").forEach((input) => {
    input.addEventListener("change", () => {
      state.referenceOverlays[input.dataset.referenceOverlay] = input.checked;
      renderCropOverlay(selectedBall());
    });
  });
}

function overlayMatrixRow(ball, row) {
  const selection = overlaySelectionFor(row.key);
  const variant = evidenceVariant(ball, row.key);
  const active = row.key === state.primaryEvidenceKey ? " active" : "";
  const backgroundClass = row.hasBackground && row.key === state.cropBackground ? " background-active" : "";
  const score = evidenceViewScore(variant);
  return `
    <tr class="overlay-row${active}${backgroundClass}">
      <td>
        <button type="button" data-evidence-row="${escapeAttr(row.key)}" class="evidence-row-button">
          <strong>${escapeHtml(row.label)}</strong>
          <small>${escapeHtml(row.short || row.sampling || "")}</small>
        </button>
      </td>
      <td class="view-score-cell" title="${escapeAttr(evidenceViewScoreTitle(score))}">
        ${formatViewScore(score)}
      </td>
      ${OVERLAY_COLUMNS.map(([columnKey]) => {
        const enabled = overlayColumnAvailable(row.key, columnKey, variant);
        return `
          <td>
            <input
              type="checkbox"
              data-overlay-row="${escapeAttr(row.key)}"
              data-overlay-column="${escapeAttr(columnKey)}"
              ${selection[columnKey] ? "checked" : ""}
              ${enabled ? "" : "disabled"}
              aria-label="${escapeAttr(row.label)} ${escapeAttr(columnKey)}">
          </td>
        `;
      }).join("")}
    </tr>
  `;
}

function selectedMapAsset(ball) {
  if (state.cropBackground === "source") return null;
  const assets = ball.evidence?.diagnostics?.evidence_maps?.assets || [];
  return assets.find((asset) => asset.key === state.cropBackground) || null;
}

function cropRasterUri(ball) {
  const asset = selectedMapAsset(ball);
  return asset?.uri || ball.evidence?.crop_uri || "";
}

function evidenceRows(ball) {
  const maps = ball.evidence?.diagnostics?.evidence_maps || {};
  const assets = (maps.assets || []).slice().sort((a, b) => {
    const ia = MAP_ASSET_ORDER.includes(a.key) ? MAP_ASSET_ORDER.indexOf(a.key) : MAP_ASSET_ORDER.length;
    const ib = MAP_ASSET_ORDER.includes(b.key) ? MAP_ASSET_ORDER.indexOf(b.key) : MAP_ASSET_ORDER.length;
    return ia - ib || String(a.key).localeCompare(String(b.key));
  });
  return [
    {
      key: "source",
      label: "Source image",
      short: "default",
      description: "Original source crop. Default boundary points come from source crop/background/color-difference evidence with edge fallback.",
      hasBackground: true,
    },
    ...assets.map((asset) => ({
      key: asset.key,
      label: asset.label,
      short: selectedVariantLabel(ball, asset.key),
      description: asset.description || "",
      hasBackground: true,
    })),
  ];
}

function selectedVariantLabel(ball, key) {
  const variant = evidenceVariant(ball, key);
  if (!variant?.sampling) return "";
  return displayLabel(variant.sampling);
}

function evidenceViewScore(variant) {
  return variant?.view_score || null;
}

function formatViewScore(score) {
  if (!score || score.score == null) return `<span class="view-score unavailable">n/a</span>`;
  const pct = Math.round(Number(score.score) * 100);
  const level = score.level || "unknown";
  return `<span class="view-score level-${escapeAttr(level)}">${pct}%</span>`;
}

function evidenceViewScoreTitle(score) {
  if (!score || score.score == null) {
    return "No view score: missing accepted points, ellipse, or physical outline.";
  }
  const rms = score.physical_rms_error_px == null ? "n/a" : `${fmt(score.physical_rms_error_px)} px`;
  const accepted = score.accepted_count ?? 0;
  const rejected = score.rejected_count ?? 0;
  return [
    `Diagnostic view score: ${Math.round(Number(score.score) * 100)}% (${score.level || "unknown"})`,
    `Physical outline RMS: ${rms}`,
    `Accepted/rejected points: ${accepted}/${rejected}`,
    score.formula || "",
  ].filter(Boolean).join(" | ");
}

function evidenceViewScoreText(score) {
  if (!score || score.score == null) return "n/a";
  const rms = score.physical_rms_error_px == null ? "n/a" : `${fmt(score.physical_rms_error_px)} px RMS`;
  return `${Math.round(Number(score.score) * 100)}% ${displayLabel(score.level || "unknown")} · ${rms}`;
}

function evidenceViewScorePlain(score) {
  if (!score || score.score == null) return "n/a";
  return `${Math.round(Number(score.score) * 100)}% ${displayLabel(score.level || "unknown")}`;
}

function ensureOverlaySelections(ball) {
  const rows = evidenceRows(ball);
  if (!state.overlaySelections) state.overlaySelections = {};
  for (const row of rows) {
    if (!state.overlaySelections[row.key]) {
      state.overlaySelections[row.key] = { points: false, rejected: false, ellipse: false, center: false };
    }
  }
  if (!Object.values(state.overlaySelections).some((selection) => (
    selection.points || selection.rejected || selection.ellipse || selection.center
  ))) {
    const key = recommendedEvidenceKey(ball);
    state.primaryEvidenceKey = key;
    const row = evidenceRows(ball).find((item) => item.key === key);
    if (row?.hasBackground) state.cropBackground = key;
    state.overlaySelections[key] = {
      points: overlayColumnAvailable(key, "points", evidenceVariant(ball, key)),
      rejected: overlayColumnAvailable(key, "rejected", evidenceVariant(ball, key)),
      ellipse: overlayColumnAvailable(key, "ellipse", evidenceVariant(ball, key)),
      center: overlayColumnAvailable(key, "center", evidenceVariant(ball, key)),
    };
  }
}

function recommendedEvidenceKey(ball) {
  const key = ball.evidence?.diagnostics?.final_image_evidence?.selected_map;
  if (key && evidenceRows(ball).some((row) => row.key === key)) return key;
  return "source";
}

function overlaySelectionFor(rowKey) {
  if (!state.overlaySelections) state.overlaySelections = {};
  if (!state.overlaySelections[rowKey]) {
    state.overlaySelections[rowKey] = { points: false, rejected: false, ellipse: false, center: false };
  }
  return state.overlaySelections[rowKey];
}

function activateEvidenceRow(ball, rowKey) {
  const row = evidenceRows(ball).find((item) => item.key === rowKey);
  if (!row) return;
  state.primaryEvidenceKey = rowKey;
  if (row.hasBackground) state.cropBackground = rowKey;
  state.overlaySelections = {};
  for (const item of evidenceRows(ball)) {
    state.overlaySelections[item.key] = { points: false, rejected: false, ellipse: false, center: false };
  }
  const variant = evidenceVariant(ball, rowKey);
  state.overlaySelections[rowKey] = {
    points: overlayColumnAvailable(rowKey, "points", variant),
    rejected: overlayColumnAvailable(rowKey, "rejected", variant),
    ellipse: overlayColumnAvailable(rowKey, "ellipse", variant),
    center: overlayColumnAvailable(rowKey, "center", variant),
  };
}

function overlayColumnAvailable(rowKey, columnKey, variant) {
  if (columnKey === "points") return (variant.points_px || []).length > 0;
  if (columnKey === "rejected") return (variant.rejected_points_px || []).length > 0;
  if (columnKey === "ellipse") return Boolean(variant.ellipse_fit?.center_px);
  if (columnKey === "center") return Boolean(variant.ellipse_fit?.center_px);
  return false;
}

function evidenceVariant(ball, rowKey) {
  if (rowKey === "source") {
    return {
      key: "source",
      label: "Source image",
      source: ball.evidence?.boundary_source || "source_boundary",
      sampling: "default source sampler",
      points_px: ball.evidence?.boundary_points_px || [],
    rejected_points_px: ball.evidence?.boundary_rejected_points_px || [],
    ellipse_fit: ball.evidence?.image_model || null,
    filter: ball.evidence?.boundary_filter || {},
    view_score: ball.evidence?.diagnostics?.source_boundary_view_score || null,
    addback_scenarios: ball.evidence?.diagnostics?.rejection_addback_scenarios || [],
    consensus_reject_refit: ball.evidence?.diagnostics?.consensus_reject_refit || null,
    arc_combination_refit: ball.evidence?.diagnostics?.arc_combination_refit || null,
  };
  }
  const variants = ball.evidence?.diagnostics?.evidence_maps?.boundary_variants || {};
  const variant = variants[rowKey];
  const finalVariant = finalPolicyEvidenceVariant(ball, rowKey, variant);
  if (finalVariant) return finalVariant;
  return variant?.status === "computed"
    ? variant
    : { key: rowKey, points_px: [], rejected_points_px: [], ellipse_fit: null, filter: {} };
}

function finalPolicyEvidenceVariant(ball, rowKey, baseVariant = null) {
  const policy = ball.evidence?.diagnostics?.final_image_evidence || {};
  if (!policy.used_for_final_position || policy.selected_map !== rowKey) return null;
  if (!policy.ellipse_fit?.center_px) return null;
  return {
    key: rowKey,
    label: policy.selected_label || baseVariant?.label || rowKey,
    description: baseVariant?.description || "",
    status: "computed",
    source: policy.observed_source || baseVariant?.source || `final_evidence_map_${rowKey}`,
    sampling: policy.sampling || baseVariant?.sampling || "",
    points_px: policy.boundary_points_px || [],
    rejected_points_px: policy.boundary_rejected_points_px || [],
    ellipse_fit: policy.ellipse_fit,
    filter: policy.filter || {},
    circle_baseline: baseVariant?.circle_baseline || null,
    view_score: baseVariant?.view_score || null,
    addback_scenarios: baseVariant?.addback_scenarios || [],
    consensus_reject_refit: baseVariant?.consensus_reject_refit || null,
    arc_combination_refit: baseVariant?.arc_combination_refit || null,
    promoted_final: true,
  };
}

function selectedBoundaryVariant(ball) {
  if (!ball || state.primaryEvidenceKey === "source") return null;
  const variants = ball.evidence?.diagnostics?.evidence_maps?.boundary_variants || {};
  const variant = variants[state.primaryEvidenceKey];
  return variant?.status === "computed" ? variant : null;
}

function selectedBoundaryPoints(ball) {
  return evidenceVariant(ball, state.primaryEvidenceKey || "source").points_px || [];
}

function selectedRejectedBoundaryPoints(ball) {
  return evidenceVariant(ball, state.primaryEvidenceKey || "source").rejected_points_px || [];
}

function selectedImageModel(ball) {
  return evidenceVariant(ball, state.primaryEvidenceKey || "source").ellipse_fit
    || ball.evidence?.image_model
    || null;
}

function imageEvidenceRows(ball) {
  const variant = evidenceVariant(ball, state.primaryEvidenceKey || "source");
  const row = evidenceRows(ball).find((item) => item.key === (state.primaryEvidenceKey || "source"));
  const model = selectedImageModel(ball) || {};
  const finalPolicy = ball.evidence?.diagnostics?.final_image_evidence || {};
  const filter = variant?.filter || ball.evidence?.boundary_filter || {};
  const maps = ball.evidence?.diagnostics?.evidence_maps || {};
  const activeColor = maps.active_color_model || maps.local_color_model || {};
  const localColor = maps.local_color_model || {};
  const globalCloth = maps.global_cloth_model || {};
  const fullTableEvidence = maps.full_table_evidence || {};
  const colorParams = maps.color_model_parameters || {};
  const mapStats = maps.maps || {};
  const acceptedCount = selectedBoundaryPoints(ball).length || model.point_count || 0;
  const rejectedCount = selectedRejectedBoundaryPoints(ball).length || 0;
  const assetCount = maps.assets?.length ?? 0;
  const selectedScore = evidenceViewScore(variant);
  const numbering = numberingInfo(ball);
  return [
    ["Canonical ID", numbering.canonical_ball_id == null ? `#${ball.ball_id}` : `#${numbering.canonical_ball_id} · ${escapeDisplay(numbering.slot || "canonical slot")}`],
    ["Raw detector ID", numbering.raw_detector_id == null ? "n/a" : `#${numbering.raw_detector_id}`],
    ["Observed shape", displayLabel(model.model_type || (model.center_px ? "edge_ellipse" : "none"))],
    ["Selected evidence view", row?.label || "Default source boundary"],
    ["Selected view score", evidenceViewScoreText(selectedScore)],
    ["Final center uses", finalCenterPolicyText(finalPolicy)],
    ["Sampling", variant?.sampling ? displayLabel(variant.sampling) : "default source sampler"],
    ["Evidence source", displayLabel(variant?.source || ball.evidence?.boundary_source || model.source || "n/a")],
    ["Accepted edge points", String(acceptedCount)],
    ["Rejected edge outliers", String(rejectedCount)],
    ["Rejected reasons", rejectedReasonSummary(filter)],
    ["Cluster arc-combo fit", arcCombinationRefitSummary(variant)],
    ["Diagnostic maps", `${assetCount} evidence-map backgrounds`],
    ["Evidence map source", displayLabel(maps.map_source || fullTableEvidence.map_source || "unknown")],
    ["Map normalization", displayLabel(maps.display_normalization || fullTableEvidence.display_normalization || "unknown")],
    ["Normalization scope", displayLabel(maps.normalization_scope || fullTableEvidence.normalization_scope || "unknown")],
    ["Filter", displayLabel(filter.status || "unknown")],
    ["Active cloth reference", displayLabel(activeColor.cloth_reference_mode || "unknown")],
    ["Active ball Lab", formatLab(activeColor.ball_lab)],
    ["Active cloth Lab", formatLab(activeColor.cloth_lab)],
    ["Active Lab/chroma separation", activeColor.separation_lab == null ? "n/a" : `ΔE ${fmt(activeColor.separation_lab)}, chroma ${fmt(activeColor.separation_chroma)}`],
    ["Active low contrast", activeColor.low_contrast == null ? "n/a" : String(activeColor.low_contrast)],
    ["Local annulus cloth Lab", formatLab(localColor.cloth_lab)],
    ["Local annulus separation", localColor.separation_lab == null ? "n/a" : `ΔE ${fmt(localColor.separation_lab)}, chroma ${fmt(localColor.separation_chroma)}`],
    ["Global cloth Lab", formatLab(globalCloth.cloth_lab)],
    ["Sample counts", `ball ${activeColor.ball_sample_count ?? "n/a"} · active cloth ${activeColor.cloth_sample_count ?? "n/a"} · local annulus ${localColor.cloth_sample_count ?? "n/a"}`],
    ["Color-model knobs", colorModelKnobsText(colorParams)],
    ["Boundary map p95", mapStats.combined_boundary_score?.p95 == null ? "n/a" : fmt(mapStats.combined_boundary_score.p95)],
    ["Ellipse axes", model.major_axis_px ? `${fmt(model.major_axis_px)} × ${fmt(model.minor_axis_px)} px` : "n/a"],
    ["Ellipse center", model.center_px ? `${fmt(model.center_px[0])}, ${fmt(model.center_px[1])} px` : "n/a"],
    ["Ellipse angle", model.angle_deg == null ? "n/a" : `${fmt(model.angle_deg)}°`],
    ["Image quality", model.quality || "unknown"],
  ];
}

function rejectedReasonSummary(filter) {
  const counts = filter?.rejected_reason_counts || {};
  const entries = Object.entries(counts);
  if (!entries.length) return "n/a";
  return entries
    .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
    .map(([reason, count]) => `${displayLabel(reason)} ${count}`)
    .join("; ");
}

function addbackScenarioSummary(variant) {
  const scenarios = variant?.addback_scenarios || [];
  if (!scenarios.length) return "n/a";
  const best = scenarios.find((scenario) => scenario.best_shape_match)
    || scenarios.find((scenario) => scenario.key !== "baseline" && scenario.ellipse_fit)
    || scenarios.find((scenario) => scenario.ellipse_fit);
  if (!best || !best.ellipse_fit) return "no fitted scenario";
  const ellipse = best.ellipse_fit;
  const comparison = best.cluster_shape_comparison || {};
  const score = comparison.score == null ? "n/a" : `${fmt(comparison.score)} shape-score`;
  const outlier = comparison.is_shape_outlier ? "outlier" : "shape ok";
  return `${best.label}: +${best.added_count} pts -> ${fmt(ellipse.major_axis_px)}×${fmt(ellipse.minor_axis_px)} @${fmt(ellipse.angle_deg)}°; ${score}; ${outlier}`;
}

function consensusRejectRefit(ball, evidenceKey) {
  const variant = evidenceVariant(ball, evidenceKey || "source");
  return variant?.consensus_reject_refit || null;
}

function arcCombinationRefit(ball, evidenceKey) {
  const variant = evidenceVariant(ball, evidenceKey || "source");
  return variant?.arc_combination_refit || null;
}

function consensusRejectRefitSummary(variant) {
  const refit = variant?.consensus_reject_refit;
  if (!refit) return "n/a";
  if (!refit.best?.ellipse_fit) return displayLabel(refit.reason || refit.status || "not computed");
  const best = refit.best;
  const ellipse = best.ellipse_fit;
  const comparison = best.cluster_shape_comparison || {};
  const improvement = best.shape_score_improvement == null ? "n/a" : `${fmt(best.shape_score_improvement)} score Δ`;
  const shapeScore = comparison.score == null ? "n/a" : `${fmt(comparison.score)} shape-score`;
  const status = displayLabel(refit.status || "diagnostic only");
  return `${status}: +${best.added_count} selected rejects -> ${fmt(ellipse.major_axis_px)}×${fmt(ellipse.minor_axis_px)} @${fmt(ellipse.angle_deg)}°; ${shapeScore}; ${improvement}`;
}

function arcCombinationRefitSummary(variant) {
  const refit = variant?.arc_combination_refit;
  if (!refit) return "n/a";
  if (!refit.best?.ellipse_fit) {
    return displayLabel(refit.reason || refit.status || "not computed");
  }
  const best = refit.best;
  const ellipse = best.ellipse_fit;
  const comparison = best.cluster_shape_comparison || {};
  const shapeScore = comparison.score == null ? "n/a" : `${fmt(comparison.score)} shape-score`;
  const improvement = best.shape_score_improvement == null ? "n/a" : `${fmt(best.shape_score_improvement)} score Δ`;
  const clusters = best.group_ids?.length
    ? `clusters ${best.group_ids.join("+")}`
    : "clusters n/a";
  const tried = refit.combination_count == null
    ? "n/a combos"
    : `${refit.combination_count}/${refit.theoretical_combination_count ?? "?"} combos`;
  return `${displayLabel(refit.status || "diagnostic only")}: ${clusters}, ${best.point_count} pts -> ${fmt(ellipse.major_axis_px)}×${fmt(ellipse.minor_axis_px)} @${fmt(ellipse.angle_deg)}°; ${shapeScore}; ${improvement}; ${tried}`;
}

function formatLab(values) {
  if (!Array.isArray(values) || values.length < 3) return "n/a";
  return `[${values.slice(0, 3).map((value) => fmt(value)).join(", ")}]`;
}

function colorModelKnobsText(params) {
  if (!params || Object.keys(params).length === 0) return "n/a";
  return [
    `mode ${displayLabel(params.cloth_reference_mode || "unknown")}`,
    `ball inner ${fmt(params.ball_inner_radius_factor)}`,
    `local annulus ${fmt(params.cloth_inner_radius_factor)}-${fmt(params.cloth_outer_radius_factor)} r`,
    `value ${fmt(params.minimum_value_for_color_model)}-${fmt(params.highlight_value_limit)}`,
    `global exclude ${fmt(params.global_cloth_exclusion_radius_factor)} r`,
    `erode ${params.global_cloth_erode_px ?? "n/a"} px`,
  ].join("; ");
}

function finalCenterPolicyText(policy) {
  if (!policy || !policy.status) return "current fallback source center";
  if (policy.used_for_final_position) {
    const center = policy.center_px ? ` @ ${fmt(policy.center_px[0])}, ${fmt(policy.center_px[1])} px` : "";
    return `${policy.selected_label || displayLabel(policy.selected_map || "selected evidence map")}${center}`;
  }
  return `fallback source center; ${policy.reason || displayLabel(policy.status)}`;
}

function physicalModelRows(ball) {
  const model = ball.evidence?.physical_model || {};
  const optimization = model.optimization || {};
  const cluster = clusterInfo(ball);
  const explanation = (model.explanation || []).join(" ");
  return [
    ["Model", displayLabel(model.model_type || "none")],
    ["Camera", `${displayLabel(model.camera_model || "unknown")}${model.approximate ? " (approximate)" : ""}`],
    ["Status", displayLabel(model.status || "unknown")],
    ["Projection mode", displayLabel(model.projection_mode || "forward")],
    ["Projected center", model.projected_center_px ? `${fmt(model.projected_center_px[0])}, ${fmt(model.projected_center_px[1])} px` : "n/a"],
    ["Residual", model.residual_px == null ? "n/a" : `${fmt(model.residual_px)} px`],
    ["Residual grade", displayLabel(model.residual_grade || "unknown")],
    ["Optimization", `${displayLabel(optimization.status || "n/a")}; move ${optimization.movement_from_initial_mm == null ? "n/a" : `${fmt(optimization.movement_from_initial_mm)} mm`}`],
    ["Optimized residual", optimization.residual_px == null ? "n/a" : `${fmt(optimization.residual_px)} px`],
    ["Scene constraints", cluster.cluster_status ? `${displayLabel(cluster.cluster_status)} cluster ${cluster.cluster_id}; ${cluster.component_size} balls; ${cluster.improvement_mm == null ? "n/a" : `${fmt(cluster.improvement_mm)} mm`} pair-distance improvement` : "no adjacent cluster"],
    ["Cluster shell", clusterShellText(cluster)],
    ["Cluster traversal", clusterTraversalText(cluster)],
    ["Cluster shape", clusterShapeText(cluster)],
    ["Cluster neighbor degree", cluster.cluster_neighbor_degree == null ? "n/a" : String(cluster.cluster_neighbor_degree)],
    ["Explanation", explanation || "Blue curve = forward projection from current estimated 3D ball center. It is not fitted to the blob unless physical optimization is enabled. Approximate camera model limits trust."],
    ["Observed source", displayLabel(model.observed_source || "n/a")],
    ["Height", model.z_mm == null ? "n/a" : `${fmt(model.z_mm)} mm`],
  ];
}

function renderBallStats() {
  const list = balls();
  $("statsSummary").textContent = `${list.length} balls`;
  $("ballStatsRows").innerHTML = list.map((ball) => {
    const confidence = ball.confidence || {};
    const score = confidence.score == null ? null : Math.round(confidence.score * 100);
    const evidence = ball.evidence?.image_model?.quality || confidence.level || "unknown";
    const selected = ball.ball_id === state.selectedId ? " selected" : "";
    return `
      <tr class="score-row${selected}" data-ball-id="${ball.ball_id}">
        <td>${ball.ball_id}</td>
        <td>${escapeHtml(ball.label || "unknown")}</td>
        <td>${score == null ? "n/a" : `${score}%`}</td>
        <td><span class="level-pill level-${escapeAttr(confidence.level || "unknown")}">${escapeHtml(confidence.level || "unknown")}</span></td>
        <td>${escapeHtml(displayLabel(evidence))}</td>
      </tr>
    `;
  }).join("");
  $("ballStatsRows").querySelectorAll("tr[data-ball-id]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedId = Number(row.getAttribute("data-ball-id"));
      syncUrl();
      render();
    });
  });
  renderSelectedSummary();
}

function renderSelectedSummary() {
  const ball = selectedBall();
  if (!ball) {
    $("selectedSummary").innerHTML = `<p class="muted">No ball selected.</p>`;
    return;
  }
  const confidence = ball.confidence || {};
  const image = selectedImageModel(ball) || {};
  const row = evidenceRows(ball).find((item) => item.key === (state.primaryEvidenceKey || "source"));
  const selectedScore = evidenceViewScore(evidenceVariant(ball, state.primaryEvidenceKey || "source"));
  const physical = ball.evidence?.physical_model || {};
  const cluster = clusterInfo(ball);
  const numbering = numberingInfo(ball);
  const mapCount = ball.evidence?.diagnostics?.evidence_maps?.assets?.length || 0;
  $("selectedSummary").innerHTML = `
    <div class="summary-title">#${ball.ball_id} ${escapeHtml(ball.label || "unknown")}</div>
    <dl>
      <dt>Score</dt><dd>${confidence.score == null ? "n/a" : `${Math.round(confidence.score * 100)}%`} · ${escapeHtml(confidence.level || "unknown")}</dd>
      <dt>Image evidence</dt><dd>${escapeHtml(row?.label || displayLabel(image.source || "n/a"))}; ${selectedBoundaryPoints(ball).length || image.point_count || 0} points; view ${escapeHtml(evidenceViewScorePlain(selectedScore))}; ${mapCount} maps</dd>
      <dt>Physical residual</dt><dd>${physical.residual_px == null ? "n/a" : `${fmt(physical.residual_px)} px`} · ${escapeHtml(displayLabel(physical.residual_grade || "unknown"))}</dd>
      <dt>Scene</dt><dd>${escapeHtml(cluster.cluster_status ? `${displayLabel(cluster.cluster_status)} adjacent cluster; ${clusterShellText(cluster)}` : "no adjacent cluster")}</dd>
      <dt>Traversal</dt><dd>${escapeHtml(clusterTraversalText(cluster))}</dd>
      <dt>Cluster shape</dt><dd>${escapeHtml(clusterShapeText(cluster))}</dd>
      <dt>Raw detector</dt><dd>${escapeHtml(numbering.raw_detector_id == null ? "n/a" : `#${numbering.raw_detector_id}`)}</dd>
      <dt>Source pixel</dt><dd>${ball.source_px ? `${fmt(ball.source_px[0])}, ${fmt(ball.source_px[1])}` : "n/a"}</dd>
    </dl>
  `;
}

function confidenceRows(ball) {
  const confidence = ball.confidence || {};
  const finalPolicy = ball.evidence?.diagnostics?.final_image_evidence || {};
  const cluster = clusterInfo(ball);
  return [
    ["Score", confidence.score == null ? "n/a" : `${Math.round(confidence.score * 100)}%`],
    ["Level", displayLabel(confidence.level || "unknown")],
    ["Method", displayLabel(confidence.method || "unknown")],
    ["Components", confidenceComponents(confidence.components || {})],
    ["Scene constraints", cluster.cluster_status ? `${displayLabel(cluster.cluster_status)}; pair RMS ${cluster.initial_pair_rms_mm == null ? "n/a" : `${fmt(cluster.initial_pair_rms_mm)}→${fmt(cluster.joint_pair_rms_mm)} mm`}` : "no adjacent cluster"],
    ["Cluster shell", clusterShellText(cluster)],
    ["Cluster traversal", clusterTraversalText(cluster)],
    ["Cluster shape", clusterShapeText(cluster)],
    ["Ground truth", "none in this score; it is image/physics/scene agreement, not measured accuracy"],
    ["Score rule", "start with legacy detector score, then use the best physical/image agreement score only when the sphere residual is not low"],
    ["Final source center", finalCenterPolicyText(finalPolicy)],
    ["Reasons", (confidence.reasons || []).map(displayLabel).join(", ") || "none"],
    ["Source pixel", ball.source_px ? `${fmt(ball.source_px[0])}, ${fmt(ball.source_px[1])}` : "n/a"],
    ["Table XY", ball.table_xy_mm ? `${fmt(ball.table_xy_mm[0])}, ${fmt(ball.table_xy_mm[1])} mm` : "n/a"],
  ];
}

function selectByOffset(offset) {
  const list = balls();
  if (!list.length) return;
  const index = list.findIndex((ball) => ball.ball_id === state.selectedId);
  const next = list[(index + offset + list.length) % list.length];
  state.selectedId = next.ball_id;
  syncUrl();
  render();
}

function syncUrl() {
  if (!state.stem) return;
  const params = new URLSearchParams();
  params.set("report", state.stem);
  if (state.selectedId != null) params.set("ball", String(state.selectedId));
  history.replaceState(null, "", `?${params.toString()}`);
}

function confidenceComponents(components) {
  const entries = Object.entries(components || {});
  if (!entries.length) return "n/a";
  return entries
    .map(([key, value]) => {
      const numeric = Number(value);
      const rendered = Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : String(value);
      return `${displayLabel(key)} ${rendered}`;
    })
    .join("; ");
}

function showLegend() {
  $("floatingLegend").classList.remove("hidden");
}

function hideLegend() {
  $("floatingLegend").classList.add("hidden");
}

function setupLegendDrag() {
  const panel = $("floatingLegend");
  const handle = $("legendHeader");
  let drag = null;
  handle.addEventListener("mousedown", (event) => {
    if (event.target?.id === "legendClose") return;
    const rect = panel.getBoundingClientRect();
    drag = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    event.preventDefault();
  });
  window.addEventListener("mousemove", (event) => {
    if (!drag) return;
    const width = panel.offsetWidth || 360;
    const height = panel.offsetHeight || 260;
    const x = clamp(event.clientX - drag.offsetX, 8, window.innerWidth - width - 8);
    const y = clamp(event.clientY - drag.offsetY, 8, window.innerHeight - height - 8);
    panel.style.left = `${x}px`;
    panel.style.top = `${y}px`;
  });
  window.addEventListener("mouseup", () => {
    drag = null;
  });
}

function renderDefinitionList(id, rows) {
  $(id).innerHTML = rows
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`)
    .join("");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function assetUrl(path) {
  return path ? `${state.assetsBase}${path}` : "";
}

function drawCross(svg, x, y, className, size = 22) {
  svg.appendChild(el("line", { x1: x - size, y1: y, x2: x + size, y2: y, class: className }));
  svg.appendChild(el("line", { x1: x, y1: y - size, x2: x, y2: y + size, class: className }));
}

function labelAt(x, y, text) {
  const group = el("g");
  const labelScale = sourceLabelScale();
  group.setAttribute("transform", `translate(${x} ${y}) scale(${labelScale})`);
  group.setAttribute("class", "source-label");
  group.setAttribute("data-x", x);
  group.setAttribute("data-y", y);
  const width = Math.max(42, 22 + String(text).length * 17);
  group.appendChild(el("rect", { x: 0, y: -30, width, height: 34, rx: 6, class: "label-bg" }));
  group.appendChild(el("text", { x: 10, y: -7, class: "label-text" }, text));
  return group;
}

function el(name, attrs = {}, text = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function displayLabel(value) {
  const labels = {
    radial_boundary: "edge boundary",
    radial_boundary_filtered: "filtered edge boundary",
    radial_edge: "edge boundary",
    radial_edge_filtered: "filtered edge boundary",
    radial_edge_ellipse: "edge ellipse",
    edge_ellipse: "edge ellipse",
    source_boundary_points_px: "source boundary pixels",
    source_mask_contour_points_px: "segmentation contour pixels",
    source_ellipse_fit: "source edge ellipse",
    projected_sphere: "projected sphere",
    approximate_pinhole_from_corners: "approximate pinhole from table corners",
    physical_model_plus_radial_edge_image_evidence: "physical model plus image evidence",
    physical_model_plus_image_evidence: "physical model plus image evidence",
    circle_radial: "circle diagnostic",
    fallback_radial: "fallback estimate",
    manual_homography: "table-corner bootstrap",
    source_refined_center_px: "final source-pixel estimate",
    candidate_c_and_sphere_match: "image evidence and physical model match",
    filtered: "outliers removed",
    optimized: "optimized",
    no_better_solution: "no better solution",
    physical_projection_band_edge_probability_search: "physical projection diagnostic",
    outward_drop: "outward transition",
    peak_response: "peak response",
    no_outliers: "no outliers",
    disabled: "not applied",
    fallback_unfiltered: "not applied; too few safe inliers",
    neighbor_ellipse_overlap: "neighbor ellipse overlap",
    cluster_shape_outlier: "cluster shape outlier",
    cluster_ellipse_size_outlier: "cluster ellipse size outlier",
    cluster_ellipse_angle_outlier: "cluster ellipse angle outlier",
    neighbor_ellipse_ownership_conflict: "neighbor ellipse ownership conflict",
    cluster_ellipse_major_outlier: "ellipse major-axis outlier",
    cluster_ellipse_minor_outlier: "ellipse minor-axis outlier",
    angular_segment_endpoint: "arc endpoint",
    local_radius_spike: "local radius spike",
    ellipse_residual_outlier: "ellipse residual outlier",
    other_rejected: "other reject",
    unknown_rejected: "unknown reject",
    shape_prior_match: "shape-prior match",
  };
  const text = String(value ?? "");
  if (labels[text]) return labels[text];
  return text
    .replaceAll("candidate_c", "image evidence")
    .replaceAll("candidate_d", "physical model")
    .replaceAll("candidate_b", "segmentation diagnostic")
    .replaceAll("candidate_a", "circle diagnostic")
    .replaceAll("manual_homography", "table-corner bootstrap")
    .replaceAll("source_refined_center_px", "final source-pixel estimate")
    .replaceAll("radial", "edge")
    .replaceAll("_", " ");
}

function clusterInfo(ball) {
  return ball?.evidence?.diagnostics?.scene_constraints?.joint_cluster
    || ball?.evidence?.physical_model?.optimization?.joint_cluster
    || {};
}

function numberingInfo(ball) {
  return ball?.evidence?.diagnostics?.ball_numbering || {};
}

function clusterShellTag(cluster) {
  if (!cluster || cluster.cluster_shell == null || !cluster.cluster_role) return "";
  return `${cluster.cluster_role === "perimeter" ? "P" : "I"}${cluster.cluster_shell}`;
}

function clusterTraversalTag(cluster) {
  if (!cluster || cluster.cluster_shell_status !== "computed") return "";
  const rank = cluster.cluster_traversal_primary_rank
    ?? cluster.cluster_traversal_rank_perimeter_walk
    ?? cluster.cluster_traversal_rank_cw;
  if (rank == null) return "";
  return `T${String(rank).padStart(2, "0")}`;
}

function clusterDisplayTag(cluster) {
  return [clusterShellTag(cluster), clusterTraversalTag(cluster)].filter(Boolean).join(" ");
}

function clusterShellText(cluster) {
  if (!cluster || !cluster.cluster_status) return "no adjacent cluster";
  if (cluster.cluster_shell_status === "computed" && cluster.cluster_shell != null) {
    const role = cluster.cluster_role === "perimeter" ? "perimeter" : "interior";
    const distance = cluster.cluster_perimeter_distance_mm == null ? "" : `; hull distance ${fmt(cluster.cluster_perimeter_distance_mm)} mm`;
    return `${role} shell ${cluster.cluster_shell}${distance}`;
  }
  if (cluster.cluster_shell_status) return displayLabel(cluster.cluster_shell_status);
  return "not classified";
}

function clusterTraversalText(cluster) {
  if (!cluster || !cluster.cluster_status) return "no adjacent cluster";
  if (cluster.cluster_traversal_status !== "computed") {
    return displayLabel(cluster.cluster_traversal_status || "not computed");
  }
  const cw = cluster.cluster_traversal_rank_cw == null ? "n/a" : `T${String(cluster.cluster_traversal_rank_cw).padStart(2, "0")}`;
  const ccw = cluster.cluster_traversal_rank_ccw == null ? "n/a" : `T${String(cluster.cluster_traversal_rank_ccw).padStart(2, "0")}`;
  const walk = cluster.cluster_traversal_rank_perimeter_walk == null
    ? "n/a"
    : `T${String(cluster.cluster_traversal_rank_perimeter_walk).padStart(2, "0")}`;
  return `diagnostic outside-in order: walk ${walk}, CW ${cw}, CCW ${ccw}`;
}

function clusterShapeText(cluster) {
  const shape = cluster?.cluster_shape_prior || {};
  if (!shape || !shape.status) return "not computed";
  if (shape.status !== "computed") return displayLabel(shape.status);
  const status = shape.is_shape_outlier ? "outlier" : "consistent";
  const axes = shape.ellipse_major_axis_px == null
    ? "axes n/a"
    : `${fmt(shape.ellipse_major_axis_px)}×${fmt(shape.ellipse_minor_axis_px)} px`;
  const consensus = shape.consensus_major_axis_px == null
    ? "consensus n/a"
    : `consensus ${fmt(shape.consensus_major_axis_px)}×${fmt(shape.consensus_minor_axis_px)} px @ ${fmt(shape.consensus_angle_deg)}°`;
  const deltas = shape.major_scale == null
    ? ""
    : `; scale ${fmt(shape.major_scale)}/${fmt(shape.minor_scale)}, angle Δ${fmt(shape.angle_delta_deg)}°`;
  const reasons = (shape.reasons || []).length
    ? `; ${shape.reasons.map(displayLabel).join(", ")}`
    : "";
  return `${status}: ${axes}; ${consensus}${deltas}${reasons}`;
}

function printClusterOrder() {
  const grouped = new Map();
  for (const ball of balls()) {
    const cluster = clusterInfo(ball);
    if (!cluster?.cluster_id) continue;
    if (!grouped.has(cluster.cluster_id)) grouped.set(cluster.cluster_id, []);
    grouped.get(cluster.cluster_id).push({ ball, cluster });
  }
  if (!grouped.size) {
    console.info(`[SnookerHelp] ${state.stem}: no adjacent-ball cluster order available`);
    return;
  }
  for (const [clusterId, rows] of grouped.entries()) {
    const sorted = rows.slice().sort((left, right) => (
      rankValue(left.cluster.cluster_traversal_rank_perimeter_walk ?? left.cluster.cluster_traversal_primary_rank)
      - rankValue(right.cluster.cluster_traversal_rank_perimeter_walk ?? right.cluster.cluster_traversal_primary_rank)
    ));
    const table = sorted.map(({ ball, cluster }) => ({
      path_rank: cluster.cluster_traversal_rank_perimeter_walk ?? cluster.cluster_traversal_primary_rank ?? null,
      ball_id: ball.ball_id,
      label: ball.label,
      raw_detector_id: numberingInfo(ball).raw_detector_id ?? null,
      role: cluster.cluster_role ?? null,
      shell: cluster.cluster_shell ?? null,
      cw_rank: cluster.cluster_traversal_rank_cw ?? null,
      ccw_rank: cluster.cluster_traversal_rank_ccw ?? null,
      angle_deg: cluster.cluster_traversal_angle_deg_from_top ?? null,
    }));
    console.group(`[SnookerHelp] ${state.stem} cluster ${clusterId} outside-in perimeter walk`);
    console.log(`canonical path: ${table.map((row) => `#${row.ball_id}`).join(" -> ")}`);
    console.table(table);
    console.groupEnd();
  }
}

function rankValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 9999;
}

function escapeDisplay(value) {
  return displayLabel(value || "");
}

function fmt(value) {
  return Number(value).toFixed(1);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value);
}
