//! The viewer window.

use std::collections::{HashMap, HashSet};
use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::DisplayArgs;
use crate::cog::{self, Cog, CogInfo, DataType, GeoTransform, TileKey};
use crate::composite::{self, Preset};
use crate::loader::{Event, Loader, TileCache};
use crate::render::{Colormap, DisplayStyle, StatsRequest, TileDraw, TileRenderer, Viewport};
use crate::spectral::{Sample, SpectralView};
use crate::stretch::{Histogram, StretchMode};
use crate::view::{self, Camera, Visible};

/// Most bytes of band data uploaded to the GPU in one frame, so a burst of
/// arriving tiles (or a composite change) doesn't stall a frame.
const UPLOAD_BYTES_PER_FRAME: u64 = 96 << 20;
/// Whole-image statistics use the finest overview with at most this many tiles.
const STATS_MAX_TILES: u32 = 64;
/// How long the view must be still before a live view stretch recomputes.
const LIVE_STRETCH_DELAY: Duration = Duration::from_millis(250);
const HISTOGRAM_BINS: u32 = 1024;
/// Zoom factor per scrolled logical point.
const SCROLL_ZOOM_SPEED: f64 = 0.0025;

// Debug outline colours.
const DRAWN_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 220, 120);
const CACHED_COLOR: egui::Color32 = egui::Color32::from_rgb(60, 200, 200);
const FETCHING_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 150, 30);
const QUEUED_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 160, 255);
const FAILED_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 60, 60);

/// Colours for levels in the debug tint, finest first.
const LEVEL_COLORS: [[f32; 3]; 8] = [
    [0.95, 0.30, 0.30],
    [0.95, 0.65, 0.20],
    [0.90, 0.90, 0.25],
    [0.35, 0.85, 0.35],
    [0.25, 0.80, 0.90],
    [0.35, 0.45, 0.95],
    [0.70, 0.40, 0.95],
    [0.95, 0.40, 0.80],
];

pub struct ViewerApp {
    url_input: String,
    state: State,
    renderer: TileRenderer,
    viewport: Viewport,
    /// From the command line, applied to the first file opened.
    initial_display: Option<DisplayArgs>,
    settings: Settings,
}

enum State {
    Idle,
    Opening { source: String, loader: Loader },
    Open(Box<Session>),
    Failed { source: String, error: String },
}

struct Settings {
    /// Above 1 draws coarser overviews: fewer tiles, softer image.
    lod_bias: f32,
    cpu_budget_mb: u32,
    gpu_budget_mb: u32,
    background: egui::Color32,
    debug: DebugSettings,
}

#[derive(Clone, Copy, PartialEq)]
struct DebugSettings {
    show_tiles: bool,
    labels: bool,
    tint_levels: bool,
    show_fallback: bool,
}

/// What the image is drawn as.
#[derive(Clone, Copy, Debug, PartialEq)]
enum DisplayMode {
    Composite([u32; 3]),
    Single(u32, Colormap),
}

impl DisplayMode {
    fn bands(self) -> Vec<u32> {
        match self {
            DisplayMode::Composite(bands) => bands.to_vec(),
            DisplayMode::Single(band, _) => vec![band],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    /// From the coarsest overview with at most `STATS_MAX_TILES` tiles.
    Image,
    /// From the tiles in view.
    View,
}

struct Stretch {
    /// `None` is manual: `low` and `high` as typed.
    mode: Option<StretchMode>,
    scope: Scope,
    /// Recompute (in view scope) whenever the view stops moving.
    live: bool,
    /// One range for all three channels, keeping their balance.
    linked: bool,
    gamma: f32,
    low: [f32; 3],
    high: [f32; 3],
    /// The latest statistics, per band.
    histograms: HashMap<u32, Histogram>,
    /// Statistics wanted (and for which bands and scope).
    dirty: bool,
    pending: Option<u64>,
}

struct Session {
    source: String,
    loader: Loader,
    cog: Arc<Cog>,
    transform: GeoTransform,
    pixel_size: f64,
    cache: TileCache,
    failed: HashMap<TileKey, String>,
    camera: Option<Camera>,
    display: DisplayMode,
    stretch: Stretch,
    presets: Vec<Preset>,
    spectral: SpectralView,
    show_spectral: bool,
    show_info: bool,
    hover: Option<Hover>,
    /// The last frame's visible tiles at the chosen level.
    visible: Visible,
    /// Tiles drawn last frame (fallbacks included), bottom first.
    drawn: Vec<TileKey>,
    view_changed_at: Instant,
    /// The view the live stretch last ran for.
    live_view: Option<(Camera, [f32; 2])>,
}

struct Hover {
    world: [f64; 2],
    sample: Option<Sample>,
}

impl ViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial_source: Option<String>,
        display: DisplayArgs,
    ) -> Result<Self, &'static str> {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .ok_or("the wgpu renderer is not available")?;
        #[cfg(windows)]
        crate::windows_dpi::defer_dpi_changes_while_dragging(cc);
        let mut app = Self {
            url_input: initial_source.unwrap_or_default(),
            state: State::Idle,
            renderer: TileRenderer::new(render_state),
            viewport: Viewport::default(),
            settings: Settings {
                lod_bias: 1.0,
                cpu_budget_mb: 2048,
                gpu_budget_mb: 1024,
                background: egui::Color32::from_gray(24),
                debug: DebugSettings {
                    show_tiles: display.debug_tiles,
                    labels: true,
                    tint_levels: false,
                    show_fallback: false,
                },
            },
            initial_display: Some(display),
        };
        if !app.url_input.trim().is_empty() {
            app.start_loading(&cc.egui_ctx);
        }
        Ok(app)
    }

    fn start_loading(&mut self, ctx: &egui::Context) {
        let source = self.url_input.trim().to_owned();
        let loader = Loader::open(source.clone(), ctx.clone());
        self.state = State::Opening { source, loader };
        self.renderer.clear();
    }

    fn poll_loader(&mut self) {
        self.state = match mem::replace(&mut self.state, State::Idle) {
            State::Opening { source, loader } => match loader.try_recv() {
                Some(Event::Opened(cog)) => {
                    let display = self.initial_display.take().unwrap_or_default();
                    State::Open(Box::new(Session::new(source, loader, cog, &display)))
                }
                Some(Event::OpenFailed(error)) => State::Failed { source, error },
                _ => State::Opening { source, loader },
            },
            State::Open(mut session) => {
                session.receive();
                State::Open(session)
            }
            other => other,
        };
    }

    fn url_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("COG:");
            let is_loading = matches!(self.state, State::Opening { .. });
            let buttons_width = 130.0;
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.url_input)
                    .hint_text("s3://bucket/image.tif, https://…/image.tif or a file path")
                    .desired_width(ui.available_width() - buttons_width),
            );
            let enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let open = ui
                .add_enabled(!is_loading, egui::Button::new("Open"))
                .clicked();
            if ui.button("Browse…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("GeoTIFF", &["tif", "tiff", "gtiff", "cog"])
                    .add_filter("All files", &["*"])
                    .pick_file()
            {
                self.url_input = path.display().to_string();
                self.start_loading(ui.ctx());
            } else if (open || enter) && !is_loading && !self.url_input.trim().is_empty() {
                self.start_loading(ui.ctx());
            }
        });
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_loader();
        egui::Panel::top("url_bar").show(ui, |ui| self.url_bar(ui));

        let State::Open(session) = &mut self.state else {
            egui::CentralPanel::default().show(ui, |ui| match &self.state {
                State::Idle => {
                    ui.label("Paste a COG path or URL above and press Open, or Browse for a file.");
                    ui.weak("Local files, http(s)://, s3://, gs:// and az:// all work.");
                }
                State::Opening { source, .. } => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Opening {source}"));
                    });
                }
                State::Failed { source, error } => {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("Couldn't open {source}"),
                    );
                    ui.label(error.as_str());
                }
                State::Open(_) => unreachable!(),
            });
            return;
        };

        session.poll_stats(&mut self.renderer);

        egui::Panel::bottom("status_bar").show(ui, |ui| session.status_bar(ui));
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(330.0)
            .min_size(260.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    session.controls(ui, &mut self.settings, &self.renderer);
                });
            });
        let multispectral = composite::is_multispectral(session.cog.info());
        if multispectral && session.show_spectral {
            egui::Panel::bottom("spectral")
                .resizable(true)
                .default_size(240.0)
                .min_size(120.0)
                .show(ui, |ui| {
                    let composite = match session.display {
                        DisplayMode::Composite(bands) => Some(bands),
                        DisplayMode::Single(band, _) => Some([band; 3]),
                    };
                    let hover = session.hover.as_ref().and_then(|h| h.sample.as_ref());
                    session
                        .spectral
                        .ui(ui, session.cog.info(), hover, composite);
                });
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                session.map(ui, &mut self.renderer, &mut self.viewport, &self.settings);
            });

        if session.stretch.pending.is_some() || session.stretch.dirty {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
    }
}

impl Session {
    fn new(source: String, loader: Loader, cog: Arc<Cog>, args: &DisplayArgs) -> Self {
        let info = cog.info();
        let data_bands = composite::data_band_count(info).max(1);
        let clamp = |band: u32| band.saturating_sub(1).min(data_bands - 1);
        let display = if let Some(band) = args.band {
            DisplayMode::Single(
                clamp(band),
                args.colormap.map(Into::into).unwrap_or_default(),
            )
        } else if let Some(bands) = args.bands {
            DisplayMode::Composite([clamp(bands[0]), clamp(bands[1]), clamp(bands[2])])
        } else {
            match composite::default_bands(info) {
                Some(bands) => DisplayMode::Composite(bands),
                None => DisplayMode::Single(0, args.colormap.map(Into::into).unwrap_or_default()),
            }
        };
        // 8-bit RGB opens unstretched; everything else at 2–98%.
        let mode = match args.stretch_mode() {
            Some(mode) => mode,
            None if info.is_rgb() => None,
            None => Some(StretchMode::default()),
        };
        let (low, high) = full_range(info.data_type);
        let transform = view::world_transform(info);
        Self {
            presets: composite::presets(info),
            source,
            loader,
            transform,
            pixel_size: view::pixel_size(&transform),
            cache: TileCache::default(),
            failed: HashMap::new(),
            camera: None,
            display,
            stretch: Stretch {
                mode,
                scope: Scope::Image,
                live: false,
                linked: info.is_rgb(),
                gamma: args.gamma.unwrap_or(1.0),
                low: [low; 3],
                high: [high; 3],
                histograms: HashMap::new(),
                dirty: true,
                pending: None,
            },
            spectral: SpectralView::default(),
            show_spectral: true,
            show_info: false,
            hover: None,
            visible: Visible::default(),
            drawn: Vec::new(),
            view_changed_at: Instant::now(),
            live_view: None,
            cog,
        }
    }

    fn info(&self) -> &CogInfo {
        self.cog.info()
    }

    /// Takes in tiles the loader has finished.
    fn receive(&mut self) {
        while let Some(event) = self.loader.try_recv() {
            match event {
                Event::Tile(tile) => {
                    self.failed.remove(&tile.key);
                    self.cache.insert(tile);
                }
                Event::TileFailed(key, error) => {
                    self.failed.insert(key, error);
                }
                Event::Opened(_) | Event::OpenFailed(_) => {}
            }
        }
    }

    fn style(&self, settings: &Settings) -> DisplayStyle {
        let background = settings.background.to_normalized_gamma_f32();
        let (bands, colormap) = match self.display {
            DisplayMode::Composite(bands) => (bands, None),
            DisplayMode::Single(band, colormap) => ([band; 3], Some(colormap)),
        };
        DisplayStyle {
            bands,
            colormap,
            low: self.stretch.low,
            high: self.stretch.high,
            gamma: self.stretch.gamma,
            background,
        }
    }

    /// The tiles whole-image statistics are computed from.
    fn stats_tiles(&self) -> (u8, Vec<TileKey>) {
        let levels = self.cog.levels();
        let level = view::stats_level(levels, STATS_MAX_TILES);
        (level, view::all_tiles(&levels[usize::from(level)]))
    }

    /// The map: input, choosing and fetching tiles, drawing, overlays.
    fn map(
        &mut self,
        ui: &mut egui::Ui,
        renderer: &mut TileRenderer,
        viewport: &mut Viewport,
        settings: &Settings,
    ) {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        let size_px = [rect.width() * ppp, rect.height() * ppp];
        if size_px[0] < 1.0 || size_px[1] < 1.0 {
            return;
        }
        let info = self.cog.info().clone();
        let mut camera = *self
            .camera
            .get_or_insert_with(|| Camera::fit(&info, size_px));
        let to_px = |pos: egui::Pos2| [(pos.x - rect.min.x) * ppp, (pos.y - rect.min.y) * ppp];
        let hover_pos = response.hover_pos();

        // Input.
        if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
        {
            let delta = response.drag_delta() * ppp;
            camera.pan([delta.x, delta.y]);
        }
        if let Some(pos) = hover_pos {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (f64::from(scroll) * SCROLL_ZOOM_SPEED).exp() * f64::from(zoom);
            if factor != 1.0 {
                camera.zoom_at(size_px, to_px(pos), factor);
            }
            if response.double_clicked() {
                camera.zoom_at(size_px, to_px(pos), 2.0);
            }
        }
        if !ui.ctx().text_edit_focused() {
            let center = [size_px[0] / 2.0, size_px[1] / 2.0];
            ui.input_mut(|input| {
                if input.consume_key(egui::Modifiers::NONE, egui::Key::F)
                    || input.consume_key(egui::Modifiers::NONE, egui::Key::Home)
                {
                    camera = Camera::fit(&info, size_px);
                }
                if input.consume_key(egui::Modifiers::NONE, egui::Key::Num1) {
                    camera.units_per_px = self.pixel_size;
                }
                if input.consume_key(egui::Modifiers::NONE, egui::Key::Plus)
                    || input.consume_key(egui::Modifiers::NONE, egui::Key::Equals)
                {
                    camera.zoom_at(size_px, center, 1.5);
                }
                if input.consume_key(egui::Modifiers::NONE, egui::Key::Minus) {
                    camera.zoom_at(size_px, center, 1.0 / 1.5);
                }
            });
        }
        if Some(camera) != self.camera {
            self.view_changed_at = Instant::now();
        }
        self.camera = Some(camera);

        // Which tiles.
        let levels = self.cog.levels();
        let level = view::choose_level(
            levels,
            self.pixel_size,
            camera.units_per_px,
            f64::from(settings.lod_bias),
        );
        let visible = view::visible_tiles(
            &self.transform,
            &levels[usize::from(level)],
            &camera,
            size_px,
        );
        let bands = self.display.bands();
        let resident = |renderer: &TileRenderer, key: TileKey| {
            bands.iter().all(|&b| renderer.has_band(key, b))
        };

        // Ask for what's missing: the coarsest overview first (it covers
        // everything quickly), whole-image statistics tiles, then the view.
        let coarsest = &levels[levels.len() - 1];
        let mut wanted: Vec<TileKey> = Vec::new();
        if coarsest.tiles_across * coarsest.tiles_down <= 16 {
            wanted.extend(view::all_tiles(coarsest));
        }
        let (stats_level, stats_tiles) = self.stats_tiles();
        let need_image_stats = self.stretch.mode.is_some() && self.stretch.scope == Scope::Image;
        if need_image_stats || self.stretch.histograms.is_empty() {
            wanted.extend(&stats_tiles);
        }
        wanted.extend(&visible.tiles);
        let mut seen = HashSet::new();
        wanted.retain(|key| {
            seen.insert(*key) && !self.cache.contains(key) && !self.failed.contains_key(key)
        });
        self.loader.want(&wanted);

        // Upload the bands the display needs, for visible tiles, their
        // fallbacks and the statistics tiles.
        let missing: Vec<TileKey> = visible
            .tiles
            .iter()
            .copied()
            .filter(|&key| !resident(renderer, key))
            .collect();
        let mut fallback = view::fallback_tiles(levels, &missing, level);
        // One level finer, for zooming out over tiles already loaded.
        if level > 0 {
            for &key in &missing {
                let finer = &levels[usize::from(level) - 1];
                fallback.extend(view::tiles_in_pixel_rect(
                    finer,
                    view::tile_pixel_rect(&levels[usize::from(level)], key),
                ));
            }
        }
        fallback.retain(|key| self.cache.contains(key));
        let mut upload_order: Vec<TileKey> = visible.tiles.clone();
        upload_order.extend(&fallback);
        if self.stretch.dirty {
            upload_order.extend(&stats_tiles);
        }
        let mut uploaded = 0u64;
        let mut deferred = false;
        for key in &upload_order {
            let Some(tile) = self.cache.get(key) else {
                continue;
            };
            for &band in &bands {
                if renderer.has_band(*key, band) {
                    continue;
                }
                if uploaded >= UPLOAD_BYTES_PER_FRAME {
                    deferred = true;
                    break;
                }
                renderer.upload_band(*key, band, [tile.width, tile.height], tile.band(band));
                uploaded += u64::from(tile.width * tile.height) * 4;
            }
        }
        if deferred {
            ui.ctx().request_repaint();
        }

        // Draw: coarse fallbacks under finer ones, the chosen level on top.
        let mut order: Vec<TileKey> = fallback
            .iter()
            .copied()
            .filter(|&key| resident(renderer, key))
            .collect();
        order.sort_by_key(|key| std::cmp::Reverse(key.level));
        order.dedup();
        order.extend(
            visible
                .tiles
                .iter()
                .copied()
                .filter(|&key| resident(renderer, key)),
        );
        let draws: Vec<TileDraw> = order
            .iter()
            .map(|&key| self.tile_draw(key, &camera, size_px, settings.debug.tint_levels))
            .collect();
        let style = self.style(settings);
        let texture = renderer.render(
            viewport,
            [size_px[0] as u32, size_px[1] as u32],
            &draws,
            &style,
        );
        ui.painter().image(
            texture,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Keep what's in use; free the rest past the budgets.
        let mut keep: HashSet<TileKey> = order.iter().copied().collect();
        keep.extend(&visible.tiles);
        keep.extend(&stats_tiles);
        if coarsest.tiles_across * coarsest.tiles_down <= 16 {
            keep.extend(view::all_tiles(coarsest));
        }
        self.cache.touch(keep.iter().copied());
        self.cache.next_frame();
        self.cache
            .evict_to_budget(u64::from(settings.cpu_budget_mb) << 20, &keep);
        renderer.evict_to_budget(u64::from(settings.gpu_budget_mb) << 20, &keep);

        // Statistics, once their tiles are on the GPU.
        self.maybe_start_stats(
            renderer,
            &visible,
            stats_level,
            &stats_tiles,
            (camera, size_px),
        );

        // Hover, pins and the debug overlay.
        self.hover = hover_pos.map(|pos| {
            let world = camera.screen_to_world(size_px, to_px(pos));
            Hover {
                world,
                sample: self.sample_at(world),
            }
        });
        if response.clicked()
            && composite::is_multispectral(&info)
            && let Some(sample) = self.hover.as_ref().and_then(|h| h.sample.clone())
        {
            self.spectral.pin(sample);
        }
        let painter = ui.painter_at(rect);
        let to_screen = |world: [f64; 2]| {
            let [x, y] = camera.world_to_screen(size_px, world);
            rect.min + egui::vec2(x / ppp, y / ppp)
        };
        for (pixel, color) in self.spectral.pinned_pixels() {
            let world = self
                .transform
                .pixel_to_crs(f64::from(pixel[0]) + 0.5, f64::from(pixel[1]) + 0.5);
            let at = to_screen(world);
            painter.circle_stroke(at, 5.0, egui::Stroke::new(2.0, color));
            painter.circle_filled(at, 1.5, color);
        }
        if settings.debug.show_tiles {
            self.debug_overlay(
                &painter,
                renderer,
                &visible,
                &order,
                settings.debug,
                &to_screen,
            );
        }

        self.visible = visible;
        self.drawn = order;
    }

    fn tile_draw(&self, key: TileKey, camera: &Camera, size_px: [f32; 2], tint: bool) -> TileDraw {
        let level = &self.cog.levels()[usize::from(key.level)];
        let corners = view::tile_world_corners(&self.transform, level, key)
            .map(|world| camera.world_to_clip(size_px, world));
        let [x0, y0, x1, y1] = view::tile_level_rect(level, key);
        let tint = if tint {
            let [r, g, b] = LEVEL_COLORS[usize::from(key.level) % LEVEL_COLORS.len()];
            [r, g, b, 0.3]
        } else {
            [0.0; 4]
        };
        TileDraw {
            key,
            corners,
            texels: [(x1 - x0) as f32, (y1 - y0) as f32],
            tint,
        }
    }

    /// Every band at a world point, from the finest cached tile there.
    fn sample_at(&self, world: [f64; 2]) -> Option<Sample> {
        let pixel = view::world_to_pixel(self.info(), world)?;
        for level in self.cog.levels() {
            let Some((key, [x, y])) = view::locate_pixel(level, pixel) else {
                continue;
            };
            if let Some(tile) = self.cache.get(&key) {
                let data_bands = composite::data_band_count(self.info()) as usize;
                return Some(Sample {
                    pixel: [pixel[0] as u32, pixel[1] as u32],
                    level: level.index,
                    values: tile.pixel(x, y).take(data_bands).collect(),
                });
            }
        }
        None
    }

    fn maybe_start_stats(
        &mut self,
        renderer: &mut TileRenderer,
        visible: &Visible,
        stats_level: u8,
        stats_tiles: &[TileKey],
        view_now: (Camera, [f32; 2]),
    ) {
        let stretch = &mut self.stretch;
        // A live view stretch reruns once the view has settled somewhere new.
        if stretch.live
            && stretch.scope == Scope::View
            && stretch.live_view_differs(self.live_view, view_now)
            && self.view_changed_at.elapsed() >= LIVE_STRETCH_DELAY
        {
            stretch.dirty = true;
        }
        if !stretch.dirty || stretch.pending.is_some() {
            return;
        }
        let bands = self.display.bands();
        let levels = self.cog.levels();
        let (tiles, clip) = match stretch.scope {
            Scope::Image => (stats_tiles.to_vec(), None),
            Scope::View => (visible.tiles.clone(), Some(visible.level_rect)),
        };
        let level = match stretch.scope {
            Scope::Image => stats_level,
            Scope::View => visible.level,
        };
        let level = &levels[usize::from(level)];
        // Wait for every tile to be on the GPU, or have failed.
        let ready = tiles.iter().all(|&key| {
            self.failed.contains_key(&key) || bands.iter().all(|&b| renderer.has_band(key, b))
        });
        if !ready || tiles.is_empty() {
            return;
        }
        let regions: Vec<(TileKey, [u32; 4])> = tiles
            .iter()
            .filter(|key| !self.failed.contains_key(key))
            .filter_map(|&key| {
                let [x0, y0, x1, y1] = view::tile_level_rect(level, key);
                let (ox, oy) = (x0, y0);
                let [mut x0, mut y0, mut x1, mut y1] = [x0, y0, x1, y1].map(f64::from);
                if let Some([cx0, cy0, cx1, cy1]) = clip {
                    x0 = x0.max(cx0.floor());
                    y0 = y0.max(cy0.floor());
                    x1 = x1.min(cx1.ceil());
                    y1 = y1.min(cy1.ceil());
                }
                (x0 < x1 && y0 < y1).then(|| {
                    (
                        key,
                        [
                            x0 as u32 - ox,
                            y0 as u32 - oy,
                            x1 as u32 - ox,
                            y1 as u32 - oy,
                        ],
                    )
                })
            })
            .collect();
        stretch.pending = Some(renderer.start_stats(StatsRequest {
            bands,
            regions,
            bins: HISTOGRAM_BINS,
        }));
        stretch.dirty = false;
        self.live_view = Some(view_now);
    }

    fn poll_stats(&mut self, renderer: &mut TileRenderer) {
        for result in renderer.poll_stats() {
            if Some(result.id) != self.stretch.pending {
                continue;
            }
            self.stretch.pending = None;
            for (band, histogram) in result.histograms {
                self.stretch.histograms.insert(band, histogram);
            }
            self.stretch.apply(self.display);
        }
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let info = self.info();
            if let Some(hover) = &self.hover {
                let [x, y] = hover.world;
                if info.geo.is_some() {
                    if info.crs.geographic {
                        ui.monospace(format!("lon {x:.6}  lat {y:.6}"));
                    } else {
                        ui.monospace(format!("x {x:.2}  y {y:.2}"));
                    }
                }
                if let Some(pixel) = view::world_to_pixel(info, hover.world) {
                    ui.separator();
                    ui.monospace(format!("col {} row {}", pixel[0] as u32, pixel[1] as u32));
                }
                if let Some(sample) = &hover.sample {
                    ui.separator();
                    let values: Vec<String> = self
                        .display
                        .bands()
                        .iter()
                        .map(|&b| format_value(sample.values.get(b as usize).copied()))
                        .collect();
                    ui.monospace(format!("value {}", values.join(" / ")));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(info.crs.label());
                if let Some(camera) = &self.camera {
                    ui.separator();
                    let scale = camera.units_per_px / self.pixel_size;
                    ui.weak(format!(
                        "level {} · {} screen px per image px",
                        self.visible.level,
                        format_zoom(1.0 / scale)
                    ));
                }
            });
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui, settings: &mut Settings, renderer: &TileRenderer) {
        let info = self.cog.info().clone();
        ui.heading(file_name(&self.source));
        ui.label(format!(
            "{} × {} px · {} band{} · {}",
            info.width,
            info.height,
            info.bands,
            if info.bands == 1 { "" } else { "s" },
            data_type_name(info.data_type),
        ));
        ui.label(format!(
            "{:?} · {} levels",
            info.compression,
            self.cog.levels().len()
        ));
        ui.label(info.crs.label());
        ui.checkbox(&mut self.show_info, "Show file details");
        if self.show_info {
            let text = cog::describe(&self.cog);
            ui.add(egui::Label::new(egui::RichText::new(text).monospace().small()).wrap());
        }
        ui.separator();

        egui::CollapsingHeader::new("Bands")
            .default_open(true)
            .show(ui, |ui| self.bands_ui(ui, &info));
        egui::CollapsingHeader::new("Contrast stretch")
            .default_open(true)
            .show(ui, |ui| self.stretch_ui(ui, &info));
        egui::CollapsingHeader::new("View")
            .default_open(true)
            .show(ui, |ui| self.view_ui(ui, settings, &info));
        egui::CollapsingHeader::new("Tile debug")
            .default_open(settings.debug.show_tiles)
            .show(ui, |ui| self.debug_ui(ui, settings, renderer));
    }

    fn bands_ui(&mut self, ui: &mut egui::Ui, info: &CogInfo) {
        let data_bands = composite::data_band_count(info);
        let before = self.display;
        ui.horizontal(|ui| {
            let is_composite = matches!(self.display, DisplayMode::Composite(_));
            if ui
                .add_enabled(
                    data_bands >= 2,
                    egui::RadioButton::new(is_composite, "RGB composite"),
                )
                .clicked()
                && !is_composite
            {
                self.display = DisplayMode::Composite(composite::default_bands(info).unwrap_or([
                    0,
                    1.min(data_bands - 1),
                    2.min(data_bands - 1),
                ]));
            }
            if ui.radio(!is_composite, "Single band").clicked() && is_composite {
                let DisplayMode::Composite(bands) = self.display else {
                    unreachable!()
                };
                self.display = DisplayMode::Single(bands[0], Colormap::Gray);
            }
        });
        match &mut self.display {
            DisplayMode::Composite(bands) => {
                if !self.presets.is_empty() {
                    let current = self.presets.iter().find(|p| p.bands == *bands);
                    egui::ComboBox::from_label("Preset")
                        .selected_text(current.map_or("Custom", |p| p.name))
                        .show_ui(ui, |ui| {
                            for preset in &self.presets {
                                ui.selectable_value(bands, preset.bands, preset.name);
                            }
                        });
                }
                let channels = [
                    ("R", egui::Color32::from_rgb(230, 80, 80)),
                    ("G", egui::Color32::from_rgb(80, 200, 80)),
                    ("B", egui::Color32::from_rgb(90, 130, 240)),
                ];
                for (channel, (name, color)) in channels.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.colored_label(*color, *name);
                        band_combo(ui, info, ("band", channel), &mut bands[channel], data_bands);
                    });
                }
            }
            DisplayMode::Single(band, colormap) => {
                ui.horizontal(|ui| {
                    ui.label("Band");
                    band_combo(ui, info, "single", band, data_bands);
                });
                egui::ComboBox::from_label("Colour map")
                    .selected_text(colormap.name())
                    .show_ui(ui, |ui| {
                        for option in Colormap::ALL {
                            ui.selectable_value(colormap, option, option.name());
                        }
                    });
            }
        }
        if composite::is_multispectral(info) {
            ui.checkbox(&mut self.show_spectral, "Spectral viewer");
        }
        if self.display.bands() != before.bands() {
            // New bands need their own statistics (kept ones are reused).
            self.stretch.dirty = true;
            self.stretch.apply(self.display);
        }
    }

    fn stretch_ui(&mut self, ui: &mut egui::Ui, info: &CogInfo) {
        let stretch = &mut self.stretch;
        let before = stretch.mode;
        let name = match stretch.mode {
            None => "Manual",
            Some(StretchMode::MinMax) => "Linear (min–max)",
            Some(StretchMode::Percentile { .. }) => "Percentile clip",
            Some(StretchMode::OptimizedLinear { .. }) => "Optimized linear",
            Some(StretchMode::StdDev { .. }) => "Standard deviation",
        };
        egui::ComboBox::from_label("Stretch")
            .selected_text(name)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut stretch.mode,
                    Some(StretchMode::MinMax),
                    "Linear (min–max)",
                );
                if !matches!(stretch.mode, Some(StretchMode::Percentile { .. })) {
                    ui.selectable_value(
                        &mut stretch.mode,
                        Some(StretchMode::default()),
                        "Percentile clip",
                    );
                }
                if !matches!(stretch.mode, Some(StretchMode::OptimizedLinear { .. })) {
                    ui.selectable_value(
                        &mut stretch.mode,
                        Some(StretchMode::OPTIMIZED_LINEAR),
                        "Optimized linear",
                    );
                }
                if !matches!(stretch.mode, Some(StretchMode::StdDev { .. })) {
                    ui.selectable_value(
                        &mut stretch.mode,
                        Some(StretchMode::StdDev { n: 2.0 }),
                        "Standard deviation",
                    );
                }
                ui.selectable_value(&mut stretch.mode, None, "Manual");
            });
        let mut params_changed = false;
        match &mut stretch.mode {
            Some(StretchMode::Percentile { low, high }) => {
                ui.horizontal(|ui| {
                    ui.label("Clip");
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(low)
                                .range(0.0..=50.0)
                                .speed(0.1)
                                .suffix("%"),
                        )
                        .changed();
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(high)
                                .range(50.0..=100.0)
                                .speed(0.1)
                                .suffix("%"),
                        )
                        .changed();
                });
            }
            Some(StretchMode::OptimizedLinear {
                low,
                high,
                low_adjust,
                high_adjust,
            }) => {
                ui.horizontal(|ui| {
                    ui.label("Percentiles");
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(low)
                                .range(0.0..=50.0)
                                .speed(0.1)
                                .suffix("%"),
                        )
                        .changed();
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(high)
                                .range(50.0..=100.0)
                                .speed(0.1)
                                .suffix("%"),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Widen by");
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(low_adjust)
                                .range(0.0..=2.0)
                                .speed(0.01),
                        )
                        .changed();
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(high_adjust)
                                .range(0.0..=2.0)
                                .speed(0.01),
                        )
                        .changed();
                });
            }
            Some(StretchMode::StdDev { n }) => {
                ui.horizontal(|ui| {
                    ui.label("Mean ±");
                    params_changed |= ui
                        .add(
                            egui::DragValue::new(n)
                                .range(0.1..=10.0)
                                .speed(0.05)
                                .suffix(" σ"),
                        )
                        .changed();
                });
            }
            Some(StretchMode::MinMax) | None => {}
        }

        ui.horizontal(|ui| {
            ui.label("From");
            let mut scope = stretch.scope;
            ui.selectable_value(&mut scope, Scope::Image, "Whole image")
                .on_hover_text("Statistics from an overview of the whole image");
            ui.selectable_value(&mut scope, Scope::View, "View")
                .on_hover_text("Statistics from the pixels in view");
            if scope != stretch.scope {
                stretch.scope = scope;
                stretch.dirty = true;
            }
        });
        ui.horizontal(|ui| {
            ui.add_enabled(
                stretch.scope == Scope::View,
                egui::Checkbox::new(&mut stretch.live, "Follow view"),
            )
            .on_hover_text("Restretch whenever the view stops moving");
            if ui.button("Restretch").clicked() {
                stretch.dirty = true;
            }
            if stretch.pending.is_some() || stretch.dirty {
                ui.spinner();
            }
        });
        let linked = stretch.linked;
        ui.checkbox(&mut stretch.linked, "Same range for R, G and B")
            .on_hover_text("Keeps the channels' balance, e.g. for true colour");
        ui.add(
            egui::Slider::new(&mut stretch.gamma, 0.2..=3.0)
                .text("Gamma")
                .logarithmic(true),
        );

        if stretch.mode != before || params_changed || stretch.linked != linked {
            stretch.apply(self.display);
        }

        // Histograms, with each channel's range, which can be typed in.
        let bands = self.display.bands();
        let channel_colors = if bands.len() == 1 {
            vec![ui.visuals().text_color()]
        } else {
            vec![
                egui::Color32::from_rgb(230, 80, 80),
                egui::Color32::from_rgb(80, 200, 80),
                egui::Color32::from_rgb(90, 130, 240),
            ]
        };
        let speed = match info.data_type {
            DataType::F32 | DataType::F64 => 0.001,
            _ => 1.0,
        };
        let mut edited = false;
        for (channel, (&band, color)) in bands.iter().zip(&channel_colors).enumerate() {
            ui.horizontal(|ui| {
                ui.colored_label(*color, composite::band_label(info, band));
            });
            if let Some(histogram) = stretch.histograms.get(&band) {
                histogram_plot(
                    ui,
                    ("hist", channel),
                    histogram,
                    *color,
                    (stretch.low[channel], stretch.high[channel]),
                );
            }
            ui.horizontal(|ui| {
                let range_speed = speed
                    * f64::from(
                        (stretch.high[channel] - stretch.low[channel])
                            .abs()
                            .max(1.0),
                    )
                    / 100.0;
                edited |= ui
                    .add(
                        egui::DragValue::new(&mut stretch.low[channel])
                            .speed(range_speed)
                            .prefix("low "),
                    )
                    .changed();
                edited |= ui
                    .add(
                        egui::DragValue::new(&mut stretch.high[channel])
                            .speed(range_speed)
                            .prefix("high "),
                    )
                    .changed();
            });
        }
        if edited {
            stretch.mode = None;
            if stretch.linked {
                let (low, high) = (stretch.low, stretch.high);
                // Copy whichever channel changed to the others.
                for channel in 0..3 {
                    stretch.low[channel] = low[0];
                    stretch.high[channel] = high[0];
                }
            }
            // A single band uses channel 0 only; keep the others alike.
            if bands.len() == 1 {
                stretch.low = [stretch.low[0]; 3];
                stretch.high = [stretch.high[0]; 3];
            }
        }
    }

    fn view_ui(&mut self, ui: &mut egui::Ui, settings: &mut Settings, info: &CogInfo) {
        ui.horizontal(|ui| {
            if ui
                .button("Fit")
                .on_hover_text("Show the whole image (F)")
                .clicked()
            {
                self.camera = None;
            }
            if ui
                .button("1:1")
                .on_hover_text("One image pixel per screen pixel (1)")
                .clicked()
                && let Some(camera) = &mut self.camera
            {
                camera.units_per_px = self.pixel_size;
            }
        });
        ui.add(
            egui::Slider::new(&mut settings.lod_bias, 0.5..=4.0)
                .logarithmic(true)
                .text("Detail bias"),
        )
        .on_hover_text("Above 1 draws coarser overviews: fewer tiles, softer image");
        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_srgba(&mut settings.background);
        });
        if let Some(geo) = &info.geo {
            ui.weak(format!(
                "Pixel size {:.3} × {:.3}{}",
                geo.0[1].abs().hypot(geo.0[4]),
                geo.0[5].abs().hypot(geo.0[2]),
                info.crs
                    .units
                    .as_deref()
                    .map(|u| format!(" {u}"))
                    .unwrap_or_default()
            ));
        } else {
            ui.weak("No georeferencing: pixel coordinates");
        }
    }

    fn debug_ui(&mut self, ui: &mut egui::Ui, settings: &mut Settings, renderer: &TileRenderer) {
        let debug = &mut settings.debug;
        ui.checkbox(&mut debug.show_tiles, "Outline tiles");
        ui.add_enabled_ui(debug.show_tiles, |ui| {
            ui.checkbox(&mut debug.labels, "Label tiles (level/x/y)");
            ui.checkbox(&mut debug.show_fallback, "Outline fallback tiles");
        });
        ui.checkbox(&mut debug.tint_levels, "Tint by overview level");
        if debug.show_tiles {
            ui.horizontal_wrapped(|ui| {
                for (color, label) in [
                    (DRAWN_COLOR, "drawn"),
                    (CACHED_COLOR, "decoded"),
                    (FETCHING_COLOR, "fetching"),
                    (QUEUED_COLOR, "queued"),
                    (FAILED_COLOR, "failed"),
                ] {
                    ui.colored_label(color, format!("■ {label}"));
                }
            });
        }

        let stats = self.loader.stats();
        let (in_flight, queued) = self.loader.snapshot();
        egui::Grid::new("stream_stats")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Fetched");
                ui.label(format!(
                    "{} tiles, {} in {} batches",
                    stats.tiles,
                    format_bytes(stats.bytes),
                    stats.batches
                ));
                ui.end_row();
                ui.label("Fetching");
                ui.label(format!("{} ({} queued)", in_flight.len(), queued.len()));
                ui.end_row();
                ui.label("Failed");
                ui.label(stats.failed.to_string());
                ui.end_row();
                ui.label("CPU cache");
                ui.label(format!(
                    "{} tiles, {}",
                    self.cache.len(),
                    format_bytes(self.cache.bytes())
                ));
                ui.end_row();
                ui.label("GPU");
                ui.label(format_bytes(renderer.resident_bytes()));
                ui.end_row();
                ui.label("Drawn");
                ui.label(format!(
                    "{} tiles at level {}",
                    self.drawn.len(),
                    self.visible.level
                ));
                ui.end_row();
            });
        if let Some((key, error)) = self.failed.iter().next() {
            ui.colored_label(ui.visuals().error_fg_color, format!("{key}: {error}"));
        }

        ui.label("Levels");
        egui::Grid::new("levels")
            .striped(true)
            .num_columns(4)
            .show(ui, |ui| {
                ui.weak("#");
                ui.weak("size");
                ui.weak("tiles");
                ui.weak("cached");
                ui.end_row();
                for level in self.cog.levels() {
                    let cached = self.cache.keys().filter(|k| k.level == level.index).count();
                    let label = format!("{}", level.index);
                    if level.index == self.visible.level {
                        ui.strong(label);
                    } else {
                        ui.label(label);
                    }
                    ui.label(format!("{}×{}", level.width, level.height));
                    ui.label(format!(
                        "{}×{} of {}²",
                        level.tiles_across, level.tiles_down, level.tile_width
                    ));
                    ui.label(cached.to_string());
                    ui.end_row();
                }
            });
        ui.add(
            egui::Slider::new(&mut settings.cpu_budget_mb, 256..=16384)
                .logarithmic(true)
                .text("CPU cache MB"),
        );
        ui.add(
            egui::Slider::new(&mut settings.gpu_budget_mb, 128..=8192)
                .logarithmic(true)
                .text("GPU cache MB"),
        );
    }

    fn debug_overlay(
        &self,
        painter: &egui::Painter,
        renderer: &TileRenderer,
        visible: &Visible,
        drawn: &[TileKey],
        debug: DebugSettings,
        to_screen: &dyn Fn([f64; 2]) -> egui::Pos2,
    ) {
        let levels = self.cog.levels();
        let (in_flight, queued) = self.loader.snapshot();
        let queued: HashSet<TileKey> = queued.into_iter().collect();
        let bands = self.display.bands();
        let outline = |key: TileKey, color: egui::Color32, width: f32, label: bool| {
            let level = &levels[usize::from(key.level)];
            let corners = view::tile_world_corners(&self.transform, level, key).map(to_screen);
            let mut points = corners.to_vec();
            points.push(corners[0]);
            painter.add(egui::Shape::line(points, egui::Stroke::new(width, color)));
            let size = corners[0].distance(corners[2]);
            if label && size > 90.0 {
                let centre = egui::pos2(
                    corners.iter().map(|p| p.x).sum::<f32>() / 4.0,
                    corners.iter().map(|p| p.y).sum::<f32>() / 4.0,
                );
                painter.text(
                    centre,
                    egui::Align2::CENTER_CENTER,
                    key.to_string(),
                    egui::FontId::monospace(12.0),
                    color,
                );
            }
        };
        if debug.show_fallback {
            for &key in drawn.iter().filter(|key| key.level != visible.level) {
                let [r, g, b] = LEVEL_COLORS[usize::from(key.level) % LEVEL_COLORS.len()];
                let color = egui::Rgba::from_rgb(r, g, b).into();
                outline(key, color, 1.0, false);
            }
        }
        for &key in &visible.tiles {
            let color = if self.failed.contains_key(&key) {
                FAILED_COLOR
            } else if bands.iter().all(|&b| renderer.has_band(key, b)) {
                DRAWN_COLOR
            } else if self.cache.contains(&key) {
                CACHED_COLOR
            } else if in_flight.contains(&key) {
                FETCHING_COLOR
            } else if queued.contains(&key) {
                QUEUED_COLOR
            } else {
                egui::Color32::GRAY
            };
            outline(key, color, 1.5, debug.labels);
        }
    }
}

impl Stretch {
    fn live_view_differs(&self, last: Option<(Camera, [f32; 2])>, now: (Camera, [f32; 2])) -> bool {
        last != Some(now)
    }

    /// Sets the ranges from the latest histograms, unless manual.
    fn apply(&mut self, display: DisplayMode) {
        let Some(mode) = self.mode else { return };
        let bands = display.bands();
        let mut ranges = Vec::new();
        for &band in &bands {
            match self.histograms.get(&band) {
                Some(histogram) if histogram.total > 0 => ranges.push(mode.range(histogram)),
                _ => return,
            }
        }
        if self.linked && ranges.len() > 1 {
            let low = ranges.iter().map(|r| r.0).fold(f32::INFINITY, f32::min);
            let high = ranges.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
            ranges.iter_mut().for_each(|r| *r = (low, high));
        }
        for channel in 0..3 {
            let (low, high) = ranges[channel.min(ranges.len() - 1)];
            self.low[channel] = low;
            self.high[channel] = high;
        }
    }
}

fn band_combo(
    ui: &mut egui::Ui,
    info: &CogInfo,
    id: impl std::hash::Hash + std::fmt::Debug,
    band: &mut u32,
    data_bands: u32,
) {
    egui::ComboBox::from_id_salt(id)
        .width(ui.available_width() - 8.0)
        .selected_text(composite::band_label(info, *band))
        .show_ui(ui, |ui| {
            for option in 0..data_bands {
                ui.selectable_value(band, option, composite::band_label(info, option));
            }
        });
}

fn histogram_plot(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    histogram: &Histogram,
    color: egui::Color32,
    (low, high): (f32, f32),
) {
    use egui_plot::{Line, Plot, PlotPoints, VLine};
    let bins = histogram.counts.len().max(1);
    let width = f64::from(histogram.max - histogram.min) / bins as f64;
    // Square root counts, so a big spike doesn't flatten the rest.
    let points: PlotPoints = histogram
        .counts
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            [
                f64::from(histogram.min) + (i as f64 + 0.5) * width,
                f64::from(count).sqrt(),
            ]
        })
        .collect();
    let marker = ui.visuals().strong_text_color();
    Plot::new(egui::Id::new(id))
        .height(70.0)
        .show_axes([true, false])
        .show_grid(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show_x(false)
        .show_y(false)
        .show(ui, |plot| {
            plot.line(Line::new("count", points).color(color).fill(0.0));
            plot.vline(VLine::new("low", f64::from(low)).color(marker));
            plot.vline(VLine::new("high", f64::from(high)).color(marker));
        });
}

/// The range a data type can hold: 0–255 for 8-bit, else its full range
/// (floats: 0–1).
fn full_range(data_type: DataType) -> (f32, f32) {
    match data_type {
        DataType::U8 => (0.0, 255.0),
        DataType::I8 => (-128.0, 127.0),
        DataType::U16 => (0.0, 65535.0),
        DataType::I16 => (-32768.0, 32767.0),
        DataType::U32 => (0.0, u32::MAX as f32),
        DataType::I32 => (i32::MIN as f32, i32::MAX as f32),
        DataType::F32 | DataType::F64 => (0.0, 1.0),
    }
}

fn data_type_name(data_type: DataType) -> &'static str {
    match data_type {
        DataType::U8 => "uint8",
        DataType::I8 => "int8",
        DataType::U16 => "uint16",
        DataType::I16 => "int16",
        DataType::U32 => "uint32",
        DataType::I32 => "int32",
        DataType::F32 => "float32",
        DataType::F64 => "float64",
    }
}

fn format_value(value: Option<f32>) -> String {
    match value {
        Some(v) if v.is_nan() => "nodata".into(),
        Some(v) if v.fract() == 0.0 && v.abs() < 1e9 => format!("{v:.0}"),
        Some(v) => format!("{v:.4}"),
        None => "–".into(),
    }
}

fn format_zoom(ratio: f64) -> String {
    if ratio >= 1.0 {
        format!("{ratio:.1}")
    } else {
        format!("1/{:.1}", 1.0 / ratio)
    }
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    let bytes = bytes as f64;
    if bytes >= 1e9 {
        format!("{:.2} GB", bytes / 1e9)
    } else if bytes >= 1e6 {
        format!("{:.1} MB", bytes / 1e6)
    } else if bytes >= 1e3 {
        format!("{:.0} kB", bytes / 1e3)
    } else {
        format!("{bytes:.0} B")
    }
}

/// The last part of a path or URL, without any query.
fn file_name(source: &str) -> &str {
    let path = source.split(['?', '#']).next().unwrap_or(source);
    path.rsplit(['/', '\\'])
        .find(|s| !s.is_empty())
        .unwrap_or(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_should_take_the_last_url_segment_without_query() {
        assert_eq!(file_name("https://x.com/a/b/TCI.tif?sig=1"), "TCI.tif");
        assert_eq!(file_name(r"C:\data\dem.tif"), "dem.tif");
        assert_eq!(file_name("s3://bucket/key/image.tif"), "image.tif");
    }

    #[test]
    fn format_value_should_say_nodata_for_nan() {
        assert_eq!(format_value(Some(f32::NAN)), "nodata");
        assert_eq!(format_value(Some(1234.0)), "1234");
        assert_eq!(format_value(Some(0.125)), "0.1250");
    }

    #[test]
    fn linked_stretch_should_use_the_widest_range() {
        let histogram = |min: f32, max: f32| Histogram {
            min,
            max,
            counts: vec![1, 1],
            total: 2,
        };
        let mut stretch = Stretch {
            mode: Some(StretchMode::MinMax),
            scope: Scope::Image,
            live: false,
            linked: true,
            gamma: 1.0,
            low: [0.0; 3],
            high: [1.0; 3],
            histograms: HashMap::from([
                (0, histogram(10.0, 50.0)),
                (1, histogram(5.0, 40.0)),
                (2, histogram(20.0, 90.0)),
            ]),
            dirty: false,
            pending: None,
        };
        stretch.apply(DisplayMode::Composite([0, 1, 2]));
        assert_eq!(stretch.low, [5.0; 3]);
        assert_eq!(stretch.high, [90.0; 3]);
        stretch.linked = false;
        stretch.apply(DisplayMode::Composite([0, 1, 2]));
        assert_eq!(stretch.low, [10.0, 5.0, 20.0]);
    }
}
