use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender, TrySendError},
};

use anyhow::{Context, Result, bail};
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_SAMPLES: usize = 320; // 20 ms, a supported WebRTC VAD frame size.
pub type Frame = [i16; FRAME_SAMPLES];

pub struct Microphone {
    pub stream: Stream,
    pub frames: Receiver<Result<Frame>>,
    pub overflowed: Arc<AtomicBool>,
}

impl Microphone {
    pub fn open() -> Result<Self> {
        let device = cpal::default_host()
            .default_input_device()
            .context("No default microphone found")?;
        let supported = device
            .default_input_config()
            .context("Could not configure the default microphone. Check the input device and microphone permissions")?;
        let config = supported.config();
        // Buffer up to ten seconds while Whisper is decoding.
        let (sender, frames) = mpsc::sync_channel(500);
        let overflowed = Arc::new(AtomicBool::new(false));
        let stream = match supported.sample_format() {
            SampleFormat::I8 => build_stream::<i8>(&device, &config, sender, &overflowed),
            SampleFormat::I16 => build_stream::<i16>(&device, &config, sender, &overflowed),
            SampleFormat::I24 => build_stream::<cpal::I24>(&device, &config, sender, &overflowed),
            SampleFormat::I32 => build_stream::<i32>(&device, &config, sender, &overflowed),
            SampleFormat::I64 => build_stream::<i64>(&device, &config, sender, &overflowed),
            SampleFormat::U8 => build_stream::<u8>(&device, &config, sender, &overflowed),
            SampleFormat::U16 => build_stream::<u16>(&device, &config, sender, &overflowed),
            SampleFormat::U24 => build_stream::<cpal::U24>(&device, &config, sender, &overflowed),
            SampleFormat::U32 => build_stream::<u32>(&device, &config, sender, &overflowed),
            SampleFormat::U64 => build_stream::<u64>(&device, &config, sender, &overflowed),
            SampleFormat::F32 => build_stream::<f32>(&device, &config, sender, &overflowed),
            SampleFormat::F64 => build_stream::<f64>(&device, &config, sender, &overflowed),
            format => bail!("Unsupported microphone sample format: {format}"),
        }
        .context("Could not open the microphone")?;

        stream.play().context("Could not start the microphone")?;
        Ok(Self {
            stream,
            frames,
            overflowed,
        })
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    sender: SyncSender<Result<Frame>>,
    overflowed: &Arc<AtomicBool>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let mut resampler = Resampler::new(config.sample_rate);
    let mut frame = [0; FRAME_SAMPLES];
    let mut position = 0;

    let error_sender = sender.clone();
    let audio_overflow = Arc::clone(overflowed);
    let error_overflow = Arc::clone(overflowed);
    device.build_input_stream(
        config,
        move |samples: &[T], _| {
            for channels in samples.chunks_exact(channels) {
                let mono = channels
                    .iter()
                    .map(|&sample| sample.to_sample::<f32>())
                    .sum::<f32>()
                    / channels.len() as f32;
                resampler.push(mono, |sample| {
                    frame[position] = sample.clamp(-1.0, 1.0).to_sample::<i16>();
                    position += 1;
                    if position == FRAME_SAMPLES {
                        if let Err(TrySendError::Full(_)) = sender.try_send(Ok(frame)) {
                            audio_overflow.store(true, Ordering::Relaxed);
                        }
                        position = 0;
                    }
                });
            }
        },
        move |error| {
            if let Err(TrySendError::Full(_)) = error_sender.try_send(Err(error.into())) {
                error_overflow.store(true, Ordering::Relaxed);
            }
        },
        None,
    )
}

struct Resampler {
    input_rate: u32,
    remaining: u32,
    weighted_sum: f32,
}

impl Resampler {
    fn new(input_rate: u32) -> Self {
        Self {
            input_rate,
            remaining: input_rate,
            weighted_sum: 0.0,
        }
    }

    fn push(&mut self, sample: f32, mut emit: impl FnMut(f32)) {
        // Each output sample is the time-weighted mean over its input interval.
        // Keep the fractional overlap across microphone callbacks.
        let mut width = SAMPLE_RATE;
        while width >= self.remaining {
            self.weighted_sum += sample * self.remaining as f32;
            emit(self.weighted_sum / self.input_rate as f32);
            width -= self.remaining;
            self.remaining = self.input_rate;
            self.weighted_sum = 0.0;
        }
        self.remaining -= width;
        self.weighted_sum += sample * width as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_preserves_duration_at_44100_hz() {
        let mut resampler = Resampler::new(44_100);
        let mut emitted = 0;
        for _ in 0..44_100 {
            resampler.push(0.5, |sample| {
                assert!((sample - 0.5).abs() < 1e-6);
                emitted += 1;
            });
        }
        assert_eq!(emitted, SAMPLE_RATE);
    }
}
