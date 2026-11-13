//! Fetching and decoding tiles off the UI thread, and the CPU tile cache.
//!
//! The UI says which tiles it wants each frame, most wanted first, and the
//! list replaces the last one: tiles that scrolled out of view before a
//! worker got to them are never fetched. Workers take a few tiles of one level
//! at a time, so the COG reader can merge their byte ranges into fewer
//! requests.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use eframe::egui;

use crate::cog::{Cog, TileData, TileKey};

/// Worker threads. Remote reads spend most of their time waiting, so this is
/// more than the core count would suggest; each worker's ranges are also
/// fetched in parallel by the source.
const WORKERS: usize = 6;
/// Tiles a worker takes at once.
const BATCH: usize = 4;

pub enum Event {
    Opened(Arc<Cog>),
    OpenFailed(String),
    Tile(Arc<TileData>),
    TileFailed(TileKey, String),
}

/// Counters for the debug view.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoaderStats {
    pub tiles: u64,
    pub failed: u64,
    pub bytes: u64,
    pub batches: u64,
}

#[derive(Default)]
struct Counters {
    tiles: AtomicU64,
    failed: AtomicU64,
    bytes: AtomicU64,
    batches: AtomicU64,
}

#[derive(Default)]
struct Queue {
    cog: Option<Arc<Cog>>,
    wanted: VecDeque<TileKey>,
    in_flight: HashSet<TileKey>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
    counters: Counters,
}

pub struct Loader {
    events: Receiver<Event>,
    shared: Arc<Shared>,
}

impl Loader {
    /// Opens `location` on a new thread, then starts the workers. Repaints
    /// `ctx` whenever there's news.
    pub fn open(location: String, ctx: egui::Context) -> Self {
        let (sender, events) = mpsc::channel();
        let shared = Arc::new(Shared {
            queue: Mutex::default(),
            ready: Condvar::new(),
            counters: Counters::default(),
        });
        let opener_shared = shared.clone();
        thread::Builder::new()
            .name("cogcat-open".into())
            .spawn(move || open_and_start(location, opener_shared, sender, ctx))
            .expect("couldn't start the loader thread");
        Self { events, shared }
    }

    /// The next event, without waiting.
    pub fn try_recv(&self) -> Option<Event> {
        self.events.try_recv().ok()
    }

    /// Replaces the queue with `keys`, most wanted first. Tiles already being
    /// fetched are left out.
    pub fn want(&self, keys: &[TileKey]) {
        let mut queue = self.shared.queue.lock().expect("loader queue poisoned");
        let in_flight = &queue.in_flight;
        let wanted: VecDeque<TileKey> = keys
            .iter()
            .copied()
            .filter(|key| !in_flight.contains(key))
            .collect();
        let changed = wanted != queue.wanted;
        queue.wanted = wanted;
        if changed {
            self.shared.ready.notify_all();
        }
    }

    pub fn is_fetching(&self, key: &TileKey) -> bool {
        self.shared
            .queue
            .lock()
            .expect("loader queue poisoned")
            .in_flight
            .contains(key)
    }

    /// Tiles being fetched, and tiles waiting.
    pub fn snapshot(&self) -> (HashSet<TileKey>, Vec<TileKey>) {
        let queue = self.shared.queue.lock().expect("loader queue poisoned");
        (
            queue.in_flight.clone(),
            queue.wanted.iter().copied().collect(),
        )
    }

    pub fn stats(&self) -> LoaderStats {
        let c = &self.shared.counters;
        LoaderStats {
            tiles: c.tiles.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            bytes: c.bytes.load(Ordering::Relaxed),
            batches: c.batches.load(Ordering::Relaxed),
        }
    }
}

impl Drop for Loader {
    /// Tells the workers to stop. They aren't joined: one may be waiting on
    /// the network, and it stops on its own once that returns.
    fn drop(&mut self) {
        if let Ok(mut queue) = self.shared.queue.lock() {
            queue.shutdown = true;
            queue.wanted.clear();
        }
        self.shared.ready.notify_all();
    }
}

fn open_and_start(
    location: String,
    shared: Arc<Shared>,
    sender: Sender<Event>,
    ctx: egui::Context,
) {
    let cog = match Cog::open(&location) {
        Ok(cog) => Arc::new(cog),
        Err(error) => {
            let _ = sender.send(Event::OpenFailed(error.to_string()));
            ctx.request_repaint();
            return;
        }
    };
    {
        let mut queue = shared.queue.lock().expect("loader queue poisoned");
        if queue.shutdown {
            return;
        }
        queue.cog = Some(cog.clone());
    }
    let _ = sender.send(Event::Opened(cog));
    ctx.request_repaint();
    for index in 0..WORKERS {
        let shared = shared.clone();
        let sender = sender.clone();
        let ctx = ctx.clone();
        thread::Builder::new()
            .name(format!("cogcat-tiles-{index}"))
            .spawn(move || work(&shared, &sender, &ctx))
            .expect("couldn't start a tile worker");
    }
}

fn work(shared: &Shared, sender: &Sender<Event>, ctx: &egui::Context) {
    loop {
        let (cog, batch) = {
            let mut queue = shared.queue.lock().expect("loader queue poisoned");
            loop {
                if queue.shutdown {
                    return;
                }
                if !queue.wanted.is_empty() {
                    break;
                }
                queue = shared.ready.wait(queue).expect("loader queue poisoned");
            }
            let batch = take_batch(&mut queue.wanted);
            queue.in_flight.extend(batch.iter().copied());
            (queue.cog.clone().expect("workers start after open"), batch)
        };

        let results = cog.read_tiles(&batch);
        shared.counters.batches.fetch_add(1, Ordering::Relaxed);
        let mut queue = shared.queue.lock().expect("loader queue poisoned");
        for (key, result) in batch.iter().zip(results) {
            queue.in_flight.remove(key);
            let event = match result {
                Ok(tile) => {
                    shared.counters.tiles.fetch_add(1, Ordering::Relaxed);
                    shared
                        .counters
                        .bytes
                        .fetch_add(tile.fetched_bytes, Ordering::Relaxed);
                    Event::Tile(Arc::new(tile))
                }
                Err(error) => {
                    shared.counters.failed.fetch_add(1, Ordering::Relaxed);
                    Event::TileFailed(*key, error.to_string())
                }
            };
            if sender.send(event).is_err() {
                return;
            }
        }
        let shutdown = queue.shutdown;
        drop(queue);
        ctx.request_repaint();
        if shutdown {
            return;
        }
    }
}

/// The first wanted tile, and up to `BATCH - 1` more of the same level from
/// the next few in line.
fn take_batch(wanted: &mut VecDeque<TileKey>) -> Vec<TileKey> {
    let Some(first) = wanted.pop_front() else {
        return Vec::new();
    };
    let mut batch = vec![first];
    let mut index = 0;
    // Only look a little way down the queue, so priorities still hold.
    while batch.len() < BATCH && index < wanted.len() && index < BATCH * 2 {
        if wanted[index].level == first.level {
            batch.push(wanted.remove(index).expect("index is in range"));
        } else {
            index += 1;
        }
    }
    batch
}

/// Decoded tiles, every band, least recently used dropped first.
#[derive(Default)]
pub struct TileCache {
    tiles: HashMap<TileKey, (Arc<TileData>, u64)>,
    bytes: u64,
    frame: u64,
}

fn tile_bytes(tile: &TileData) -> u64 {
    (tile.data.len() * size_of::<f32>()) as u64
}

impl TileCache {
    pub fn insert(&mut self, tile: Arc<TileData>) {
        self.bytes += tile_bytes(&tile);
        if let Some((old, _)) = self.tiles.insert(tile.key, (tile, self.frame)) {
            self.bytes -= tile_bytes(&old);
        }
    }

    pub fn get(&self, key: &TileKey) -> Option<&Arc<TileData>> {
        self.tiles.get(key).map(|(tile, _)| tile)
    }

    pub fn contains(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    /// Marks tiles as used this frame.
    pub fn touch(&mut self, keys: impl IntoIterator<Item = TileKey>) {
        for key in keys {
            if let Some(entry) = self.tiles.get_mut(&key) {
                entry.1 = self.frame;
            }
        }
    }

    pub fn next_frame(&mut self) {
        self.frame += 1;
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.tiles.keys()
    }

    /// Drops least recently used tiles until at most `max_bytes` remain,
    /// keeping any in `keep`.
    pub fn evict_to_budget(&mut self, max_bytes: u64, keep: &HashSet<TileKey>) {
        if self.bytes <= max_bytes {
            return;
        }
        let mut candidates: Vec<(u64, TileKey)> = self
            .tiles
            .iter()
            .filter(|(key, _)| !keep.contains(key))
            .map(|(key, (_, used))| (*used, *key))
            .collect();
        candidates.sort_unstable();
        for (_, key) in candidates {
            if self.bytes <= max_bytes {
                break;
            }
            if let Some((tile, _)) = self.tiles.remove(&key) {
                self.bytes -= tile_bytes(&tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(level: u8, x: u32) -> TileKey {
        TileKey { level, x, y: 0 }
    }

    fn tile(key: TileKey) -> Arc<TileData> {
        Arc::new(TileData {
            key,
            width: 2,
            height: 2,
            valid: [2, 2],
            bands: 1,
            data: vec![0.0; 4],
            fetched_bytes: 0,
        })
    }

    #[test]
    fn take_batch_should_group_tiles_of_the_first_tiles_level() {
        let mut wanted: VecDeque<_> =
            [key(2, 0), key(0, 0), key(2, 1), key(0, 1), key(2, 2)].into();
        assert_eq!(
            take_batch(&mut wanted),
            vec![key(2, 0), key(2, 1), key(2, 2)]
        );
        assert_eq!(wanted, VecDeque::from([key(0, 0), key(0, 1)]));
    }

    #[test]
    fn take_batch_should_stop_at_the_batch_size() {
        let mut wanted: VecDeque<_> = (0..10).map(|x| key(0, x)).collect();
        assert_eq!(take_batch(&mut wanted).len(), BATCH);
        assert_eq!(wanted.len(), 10 - BATCH);
    }

    #[test]
    fn cache_should_evict_least_recently_used_tiles_first() {
        let mut cache = TileCache::default();
        cache.insert(tile(key(0, 0)));
        cache.next_frame();
        cache.insert(tile(key(0, 1)));
        cache.next_frame();
        cache.touch([key(0, 0)]);
        cache.insert(tile(key(0, 2)));
        // Each tile is 16 bytes; room for two.
        cache.evict_to_budget(32, &HashSet::new());
        assert!(!cache.contains(&key(0, 1)));
        assert!(cache.contains(&key(0, 0)) && cache.contains(&key(0, 2)));
        assert_eq!(cache.bytes(), 32);
    }

    #[test]
    fn cache_should_keep_tiles_in_use_over_budget() {
        let mut cache = TileCache::default();
        cache.insert(tile(key(0, 0)));
        cache.evict_to_budget(0, &HashSet::from([key(0, 0)]));
        assert_eq!(cache.len(), 1);
    }
}
