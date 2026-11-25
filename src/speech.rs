use std::{collections::VecDeque, mem, path::Path, thread};

use anyhow::{Context, Result, ensure};
use webrtc_vad::{SampleRate, Vad, VadMode};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::audio::{Frame, SAMPLE_RATE};

const PRE_ROLL_FRAMES: usize = 10;
const START_VOICE_FRAMES: usize = 3;
const END_SILENCE_FRAMES: usize = 20;
// Require 200 ms of VAD-positive audio before decoding. A short impulse can
// trigger several positive frames because WebRTC VAD briefly holds its decision.
const MIN_VOICED_FRAMES: usize = 10;

pub struct Segmenter {
    vad: Vad,
    pre_roll: VecDeque<Frame>,
    segment: Vec<f32>,
    max_segment_samples: usize,
    consecutive_voice: usize,
    consecutive_silence: usize,
    voiced_frames: usize,
}

impl Segmenter {
    pub fn new(max_segment_seconds: u8) -> Self {
        Self {
            vad: Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, VadMode::Aggressive),
            pre_roll: VecDeque::with_capacity(PRE_ROLL_FRAMES),
            segment: Vec::new(),
            max_segment_samples: usize::from(max_segment_seconds) * SAMPLE_RATE as usize,
            consecutive_voice: 0,
            consecutive_silence: 0,
            voiced_frames: 0,
        }
    }

    pub fn push(&mut self, frame: Frame) -> Result<Option<Vec<f32>>> {
        let voiced = self.vad.is_voice_segment(&frame).map_err(|()| {
            anyhow::anyhow!("Voice activity detection rejected a microphone frame")
        })?;
        Ok(self.push_frame(frame, voiced))
    }

    fn push_frame(&mut self, frame: Frame, voiced: bool) -> Option<Vec<f32>> {
        if self.segment.is_empty() {
            if self.pre_roll.len() == PRE_ROLL_FRAMES {
                self.pre_roll.pop_front();
            }
            self.pre_roll.push_back(frame);
            self.consecutive_voice = if voiced {
                self.consecutive_voice + 1
            } else {
                0
            };

            if self.consecutive_voice == START_VOICE_FRAMES {
                self.voiced_frames = self.consecutive_voice;
                self.segment.reserve(self.max_segment_samples);
                for frame in self.pre_roll.drain(..) {
                    append_frame(&mut self.segment, frame);
                }
            }
            return None;
        }

        append_frame(&mut self.segment, frame);
        self.voiced_frames += usize::from(voiced);
        self.consecutive_silence = if voiced {
            0
        } else {
            self.consecutive_silence + 1
        };
        if self.consecutive_silence == END_SILENCE_FRAMES
            || self.segment.len() >= self.max_segment_samples
        {
            return self.finish();
        }
        None
    }

    pub fn finish(&mut self) -> Option<Vec<f32>> {
        self.pre_roll.clear();
        self.consecutive_voice = 0;
        self.consecutive_silence = 0;
        if mem::take(&mut self.voiced_frames) < MIN_VOICED_FRAMES {
            self.segment.clear();
            return None;
        }
        Some(mem::take(&mut self.segment))
    }
}

fn append_frame(segment: &mut Vec<f32>, frame: Frame) {
    segment.extend(frame.map(|sample| f32::from(sample) / 32_768.0));
}

pub struct Recognizer {
    state: WhisperState,
    params: FullParams<'static, 'static>,
}

impl Recognizer {
    pub fn open(model: &Path, language: Option<&'static str>, beam_size: i32) -> Result<Self> {
        whisper_rs::install_logging_hooks();
        let context =
            WhisperContext::new_with_params(model, WhisperContextParameters::default())
                .with_context(|| format!("Could not load Whisper model at {}", model.display()))?;
        ensure!(
            context.is_multilingual(),
            "Use a multilingual Whisper model, such as ggml-base.bin"
        );
        let state = context
            .create_state()
            .context("Could not initialize Whisper")?;
        let threads = thread::available_parallelism().map_or(1, |count| count.get().min(4) as i32);
        let strategy = if beam_size == 1 {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(threads);
        params.set_language(language);
        params.set_no_context(true);
        params.set_no_timestamps(true);
        params.set_suppress_nst(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        Ok(Self { state, params })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        self.state
            .full(self.params.clone(), samples)
            .context("Whisper transcription failed")?;

        let mut transcript = String::new();
        for segment in self.state.as_iter() {
            transcript.push_str(
                &segment
                    .to_str_lossy()
                    .context("Could not read Whisper output")?,
            );
        }
        Ok(transcript.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::FRAME_SAMPLES;

    #[test]
    fn segmentation_keeps_pre_roll_and_flushes_after_a_pause() {
        let mut segmenter = Segmenter::new(4);
        let frame = [1; FRAME_SAMPLES];
        for _ in 0..PRE_ROLL_FRAMES * 2 {
            assert!(segmenter.push_frame(frame, false).is_none());
        }
        for _ in 0..MIN_VOICED_FRAMES {
            assert!(segmenter.push_frame(frame, true).is_none());
        }
        for _ in 1..END_SILENCE_FRAMES {
            assert!(segmenter.push_frame(frame, false).is_none());
        }
        let segment = segmenter.push_frame(frame, false).unwrap();
        assert_eq!(
            segment.len(),
            (PRE_ROLL_FRAMES + MIN_VOICED_FRAMES - START_VOICE_FRAMES + END_SILENCE_FRAMES)
                * FRAME_SAMPLES
        );
        assert!(segmenter.finish().is_none());
    }

    #[test]
    fn segmentation_flushes_pending_speech_on_shutdown() {
        let mut segmenter = Segmenter::new(4);
        for _ in 0..MIN_VOICED_FRAMES {
            segmenter.push_frame([1; FRAME_SAMPLES], true);
        }
        assert_eq!(
            segmenter.finish().unwrap().len(),
            MIN_VOICED_FRAMES * FRAME_SAMPLES
        );
        assert!(segmenter.finish().is_none());
    }

    #[test]
    fn segmentation_discards_short_noise_at_a_pause_and_at_shutdown() {
        let mut segmenter = Segmenter::new(4);
        for _ in 0..MIN_VOICED_FRAMES - 1 {
            assert!(segmenter.push_frame([1; FRAME_SAMPLES], true).is_none());
        }
        for _ in 0..END_SILENCE_FRAMES {
            assert!(segmenter.push_frame([0; FRAME_SAMPLES], false).is_none());
        }
        for _ in 0..START_VOICE_FRAMES {
            assert!(segmenter.push_frame([1; FRAME_SAMPLES], true).is_none());
        }
        assert!(segmenter.finish().is_none());
    }

    #[test]
    fn segmentation_splits_continuous_speech_at_the_configured_limit() {
        let mut segmenter = Segmenter::new(2);
        let frame = [1; FRAME_SAMPLES];
        let frames_in_segment = 2 * SAMPLE_RATE as usize / FRAME_SAMPLES;
        for _ in 1..frames_in_segment {
            assert!(segmenter.push_frame(frame, true).is_none());
        }
        let segment = segmenter.push_frame(frame, true).unwrap();
        assert_eq!(segment.len(), 2 * SAMPLE_RATE as usize);
    }
}
