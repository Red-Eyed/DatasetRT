use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub(super) struct WriteProgress {
    bar: Option<ProgressBar>,
    started_at: Instant,
    items: u64,
    bytes: u64,
    clear_on_finish: bool,
}

pub(super) struct SourceListProgress {
    multi: Option<MultiProgress>,
    bar: Option<ProgressBar>,
    started_at: Instant,
    completed: u64,
    total: u64,
}

impl WriteProgress {
    pub(super) fn new(
        enabled: bool,
        source_name: &str,
        multi_progress: Option<&MultiProgress>,
    ) -> Self {
        let clear_on_finish = multi_progress.is_some();
        let bar = enabled.then(|| create_progress_bar(source_name, multi_progress));
        Self {
            bar,
            started_at: Instant::now(),
            items: 0,
            bytes: 0,
            clear_on_finish,
        }
    }

    pub(super) fn record_sample(&mut self, byte_len: usize) {
        self.items = self.items.saturating_add(1);
        let byte_len = u64::try_from(byte_len).unwrap_or(u64::MAX);
        self.bytes = self.bytes.saturating_add(byte_len);

        if let Some(bar) = &self.bar {
            bar.inc(1);
            bar.set_message(self.message());
        }
    }

    pub(super) fn finish(&self) {
        if let Some(bar) = &self.bar {
            if self.clear_on_finish {
                bar.finish_and_clear();
            } else {
                bar.finish_with_message(self.message());
            }
        }
    }

    fn message(&self) -> String {
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64().max(0.001);
        let items_per_second = self.items as f64 / elapsed_seconds;
        let megabytes_per_second = self.bytes as f64 / 1_000_000.0 / elapsed_seconds;
        format!(
            "{} items | {:.1} items/s | {:.1} MB/s",
            self.items, items_per_second, megabytes_per_second
        )
    }
}

impl SourceListProgress {
    pub(super) fn new(enabled: bool, total: usize) -> Self {
        let total = u64::try_from(total).unwrap_or(u64::MAX);
        let multi = enabled.then(MultiProgress::new);
        let bar = multi
            .as_ref()
            .map(|multi| create_source_list_progress_bar(multi, total));
        let progress = Self {
            multi,
            bar,
            started_at: Instant::now(),
            completed: 0,
            total,
        };
        progress.refresh();
        progress
    }

    pub(super) fn multi_progress(&self) -> Option<&MultiProgress> {
        self.multi.as_ref()
    }

    pub(super) fn record_source(&mut self) {
        self.completed = self.completed.saturating_add(1);
        if let Some(bar) = &self.bar {
            bar.inc(1);
        }
        self.refresh();
    }

    pub(super) fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_with_message(self.message());
        }
    }

    fn refresh(&self) {
        if let Some(bar) = &self.bar {
            bar.set_message(self.message());
            bar.tick();
        }
    }

    fn message(&self) -> String {
        if self.completed == 0 {
            return "ETA --".to_string();
        }
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64().max(0.001);
        let sources_per_second = self.completed as f64 / elapsed_seconds;
        let remaining = self.total.saturating_sub(self.completed) as f64;
        let eta = Duration::from_secs_f64(remaining / sources_per_second);
        format!(
            "{:.1} sources/s | ETA {}",
            sources_per_second,
            format_eta(eta)
        )
    }
}

fn create_source_list_progress_bar(multi_progress: &MultiProgress, total: u64) -> ProgressBar {
    let bar = multi_progress.add(ProgressBar::new(total));
    bar.set_prefix("sources");
    if let Ok(style) =
        ProgressStyle::with_template("{wide_bar:.cyan/blue} {prefix} {pos}/{len} {msg}")
    {
        bar.set_style(style.progress_chars("=>-"));
    }
    bar
}

fn create_progress_bar(source_name: &str, multi_progress: Option<&MultiProgress>) -> ProgressBar {
    let bar = match multi_progress {
        Some(multi_progress) => multi_progress.add(ProgressBar::new_spinner()),
        None => ProgressBar::new_spinner(),
    };
    bar.set_prefix(format!("writing {source_name}"));
    bar.enable_steady_tick(Duration::from_millis(100));
    // Progress rendering is observability only; an invalid style must never fail cache writes.
    if let Ok(style) = ProgressStyle::with_template("{spinner:.green} {prefix} {msg}") {
        bar.set_style(style);
    }
    bar
}

fn format_eta(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        return format!("{seconds}s");
    }
    if seconds < 3_600 {
        return format!("{}m {:02}s", seconds / 60, seconds % 60);
    }
    format!("{}h {:02}m", seconds / 3_600, seconds % 3_600 / 60)
}
