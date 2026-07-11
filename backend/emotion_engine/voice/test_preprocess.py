import pytest
import numpy as np
import librosa
from backend.emotion_engine.voice.preprocess import (
    AudioPreprocessor,
    PeakNormalizer,
    AudioPreprocessingError
)

def test_convert_to_mono():
    preprocessor = AudioPreprocessor(mono_conversion_enabled=True)
    
    # 1. Test 1D mono audio remains unchanged
    mono_input = np.ones(1000, dtype=np.float32)
    output, converted = preprocessor.convert_to_mono(mono_input)
    assert not converted
    assert output.ndim == 1
    assert np.array_equal(output, mono_input)
    
    # 2. Test 2D stereo audio is converted to mono (by averaging channels)
    # Channel 1: all 1.0, Channel 2: all 3.0. Average should be 2.0
    stereo_input = np.array([np.ones(1000), np.ones(1000) * 3.0], dtype=np.float32)
    output, converted = preprocessor.convert_to_mono(stereo_input)
    assert converted
    assert output.ndim == 1
    assert len(output) == 1000
    assert np.allclose(output, 2.0)
    
    # 3. Test mono conversion disabled
    disabled_preprocessor = AudioPreprocessor(mono_conversion_enabled=False)
    output, converted = disabled_preprocessor.convert_to_mono(stereo_input)
    assert not converted
    assert np.array_equal(output, stereo_input)


def test_resample_audio():
    preprocessor = AudioPreprocessor()
    
    # 1. Test resampling when rates are equal (should be no-op)
    waveform = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
    output = preprocessor.resample_audio(waveform, 16000, 16000)
    assert np.array_equal(output, waveform)
    
    # 2. Test downsampling from 16000 to 8000
    # Create 1 second of audio at 16000 Hz
    t = np.linspace(0, 1, 16000, endpoint=False)
    waveform_16k = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    output_8k = preprocessor.resample_audio(waveform_16k, 16000, 8000)
    assert len(output_8k) == 8000


def test_trim_silence():
    # Set top_db = 20 for easier silence thresholding with synthetic data
    preprocessor = AudioPreprocessor(silence_threshold_db=20.0)
    
    # 1. Create a signal with leading and trailing silence (zeros)
    # Total samples: 3000. 1000 zeros, 1000 of 1.0 sin wave, 1000 zeros
    middle_signal = np.sin(np.linspace(0, 10 * np.pi, 1000)).astype(np.float32)
    waveform = np.concatenate([np.zeros(1000), middle_signal, np.zeros(1000)]).astype(np.float32)
    
    output, trimmed = preprocessor.trim_silence(waveform, top_db=20.0)
    assert trimmed
    # The output should be shorter than original waveform
    assert len(output) < len(waveform)
    
    # 2. Test with silence trimming disabled
    disabled_preprocessor = AudioPreprocessor(silence_trimming_enabled=False)
    output, trimmed = disabled_preprocessor.trim_silence(waveform, top_db=20.0)
    assert not trimmed
    assert len(output) == len(waveform)


def test_normalize_audio():
    # 1. Test peak normalizer scales the maximum amplitude to target_peak (default 1.0)
    normalizer = PeakNormalizer(target_peak=1.0)
    waveform = np.array([-0.5, 0.2, 0.5, 0.0], dtype=np.float32)
    normalized = normalizer.normalize(waveform)
    assert np.max(np.abs(normalized)) == pytest.approx(1.0)
    assert np.array_equal(normalized, np.array([-1.0, 0.4, 1.0, 0.0], dtype=np.float32))
    
    # 2. Test empty or zero array does not cause DivisionByZero error
    zero_waveform = np.zeros(100, dtype=np.float32)
    normalized_zeros = normalizer.normalize(zero_waveform)
    assert np.max(np.abs(normalized_zeros)) == 0.0


def test_pad_or_trim_audio():
    preprocessor = AudioPreprocessor()
    sr = 16000
    target_duration = 3.0  # 48000 samples
    
    # 1. Test padding: input is too short
    short_waveform = np.ones(16000, dtype=np.float32)  # 1 second
    output, padded, trimmed = preprocessor.pad_or_trim_audio(short_waveform, sr, target_duration)
    assert padded
    assert not trimmed
    assert len(output) == 48000
    assert np.array_equal(output[:16000], short_waveform)
    assert np.all(output[16000:] == 0.0)
    
    # 2. Test trimming: input is too long
    long_waveform = np.ones(60000, dtype=np.float32)  # 3.75 seconds
    output, padded, trimmed = preprocessor.pad_or_trim_audio(long_waveform, sr, target_duration)
    assert not padded
    assert trimmed
    assert len(output) == 48000
    assert np.array_equal(output, long_waveform[:48000])
    
    # 3. Test exact match: input is exactly target length
    exact_waveform = np.ones(48000, dtype=np.float32)
    output, padded, trimmed = preprocessor.pad_or_trim_audio(exact_waveform, sr, target_duration)
    assert not padded
    assert not trimmed
    assert len(output) == 48000
    assert np.array_equal(output, exact_waveform)
