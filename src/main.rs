mod audio;
mod speech;

use std::{
    fs::OpenOptions,
    io::{self, Write},
    num::NonZeroU64,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::RecvTimeoutError,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};

use crate::{
    audio::Microphone,
    speech::{Recognizer, Segmenter},
};

const POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Parser)]
#[command(
    version,
    about = "Transcribe the default microphone locally with Whisper"
)]
struct Args {
    /// Path to a multilingual Whisper GGML model.
    #[arg(long, default_value = "models/ggml-base.bin")]
    model: PathBuf,

    /// Append transcript lines to this file.
    #[arg(long, default_value = "transcript.log")]
    log_file: PathBuf,

    /// Spoken language. Auto detects the language of each segment.
    #[arg(long, value_enum, default_value = "auto")]
    language: Language,

    /// Force a split after this many seconds of continuous speech.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=20))]
    max_segment_seconds: u8,

    /// Number of decoding candidates to consider (1 uses greedy decoding).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(i32).range(1..=8))]
    beam_size: i32,

    /// Listen until interrupted, or stop after a period without recognized words.
    #[arg(long, value_enum, default_value = "forever")]
    duration: RunDuration,

    /// Seconds without recognized words before stopping in auto mode.
    #[arg(long, default_value = "30")]
    timeout: NonZeroU64,
}

#[derive(Clone, Copy, ValueEnum)]
enum Language {
    Auto,
    En,
    Zh,
}

impl Language {
    fn whisper_code(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::En => Some("en"),
            Self::Zh => Some("zh"),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum RunDuration {
    Forever,
    Auto,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stopped = Arc::new(AtomicBool::new(false));
    let interrupted = Arc::clone(&stopped);
    ctrlc::set_handler(move || interrupted.store(true, Ordering::Relaxed))
        .context("Could not install the Ctrl+C handler")?;

    ensure!(
        args.model.is_file(),
        "Whisper model not found at {}. Download a multilingual GGML model and pass its path with --model.",
        args.model.display()
    );
    let mut recognizer =
        Recognizer::open(&args.model, args.language.whisper_code(), args.beam_size)?;
    if stopped.load(Ordering::Relaxed) {
        return Ok(());
    }

    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.log_file)
        .with_context(|| {
            format!(
                "Could not open transcript log at {}",
                args.log_file.display()
            )
        })?;
    ensure!(
        same_file::Handle::from_path(&args.model)?
            != same_file::Handle::from_file(log.try_clone()?)?,
        "The transcript log and Whisper model must use different files"
    );
    let mut stdout = io::stdout().lock();
    let mut transcribe = |samples: &[f32]| -> Result<bool> {
        let transcript = recognizer.transcribe(samples)?;
        if !transcript.chars().any(char::is_alphanumeric) {
            return Ok(false);
        }
        let line = format!("{transcript}\n");
        log.write_all(line.as_bytes())
            .context("Could not append to the transcript log")?;
        stdout
            .write_all(line.as_bytes())
            .context("Could not write the transcript to stdout")?;
        stdout.flush().context("Could not flush stdout")?;
        Ok(true)
    };

    let Microphone {
        stream,
        frames,
        overflowed,
    } = Microphone::open()?;
    eprintln!("Listening on the default microphone. Press Ctrl+C to stop.");
    let mut segmenter = Segmenter::new(args.max_segment_seconds);
    let timeout = match args.duration {
        RunDuration::Forever => None,
        RunDuration::Auto => Some(Duration::from_secs(args.timeout.get())),
    };
    let mut last_words = Instant::now();

    while !stopped.load(Ordering::Relaxed) && !overflowed.load(Ordering::Relaxed) {
        let wait = timeout.map_or(POLL_INTERVAL, |limit| {
            POLL_INTERVAL.min(limit.saturating_sub(last_words.elapsed()))
        });

        match frames.recv_timeout(wait) {
            Ok(frame) => {
                if let Some(samples) = segmenter.push(frame?)?
                    && transcribe(&samples)?
                {
                    last_words = Instant::now();
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if timeout.is_some_and(|limit| last_words.elapsed() >= limit) {
                    // Give the current utterance a chance to produce words before stopping.
                    if let Some(samples) = segmenter.finish()
                        && transcribe(&samples)?
                    {
                        last_words = Instant::now();
                        continue;
                    }
                    if overflowed.load(Ordering::Relaxed) || stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    eprintln!(
                        "Stopped after {} seconds without recognized words.",
                        args.timeout
                    );
                    return Ok(());
                }
            }
            Err(RecvTimeoutError::Disconnected) => bail!("The microphone stopped sending audio"),
        }
    }

    drop(stream);
    if overflowed.load(Ordering::Relaxed) {
        eprintln!("The microphone buffer filled. Finishing buffered audio before exit.");
    }
    for frame in frames.try_iter() {
        if let Some(samples) = segmenter.push(frame?)? {
            transcribe(&samples)?;
        }
    }
    if let Some(samples) = segmenter.finish() {
        transcribe(&samples)?;
    }
    ensure!(
        !overflowed.load(Ordering::Relaxed),
        "Transcription could not keep up with the microphone. Try --beam-size 1 or a lighter model."
    );

    Ok(())
}
