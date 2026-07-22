import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from backend.emotion_engine.voice.loader import (
    VoiceEngineError,
    AudioLoadError,
    MetadataParseError,
    DatasetDetectionError,
    VoiceSample,
    VoiceDatasetLoader,
)


@pytest.fixture
def loader():
    return VoiceDatasetLoader()


@pytest.fixture
def mock_wav_dir(tmp_path):
    """Creates a minimal mock dataset directory with valid WAV files."""
    # RAVDESS structure
    ravdess_dir = tmp_path / "RAVDESS" / "Actor_01"
    ravdess_dir.mkdir(parents=True)
    for i in range(1, 4):
        wav_path = ravdess_dir / f"03-01-0{i}-01-01-01-01.wav"
        sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    # CREMA-D structure
    cremad_dir = tmp_path / "CREMA-D"
    cremad_dir.mkdir()
    for emo_code in ["HAP", "SAD", "NEU"]:
        wav_path = cremad_dir / f"1001_DFA_{emo_code}_01.wav"
        sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    return tmp_path


class TestVoiceDatasetLoader:
    def test_scan_dataset_valid(self, loader, mock_wav_dir):
        """scan_dataset returns all WAV files recursively."""
        wav_files = loader.scan_dataset(mock_wav_dir / "RAVDESS")
        assert len(wav_files) == 3
        assert all(p.suffix == ".wav" for p in wav_files)

    def test_scan_dataset_nonexistent_path(self, loader, tmp_path):
        """scan_dataset raises ValueError for missing directory."""
        with pytest.raises(ValueError, match="does not exist"):
            loader.scan_dataset(tmp_path / "nonexistent")

    def test_scan_dataset_not_a_directory(self, loader, tmp_path):
        """scan_dataset raises ValueError if path is a file, not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            loader.scan_dataset(file_path)

    def test_parse_metadata_ravdess(self, loader):
        """parse_metadata correctly parses RAVDESS filename conventions."""
        path = Path("RAVDESS/Actor_01/03-01-03-01-02-01-01.wav")
        dataset_name, emotion, speaker_id = loader.parse_metadata(path)
        assert dataset_name == "RAVDESS"
        assert emotion == "happy"
        assert speaker_id == "01"

    def test_parse_metadata_ravdess_all_emotions(self, loader):
        """parse_metadata handles all 8 RAVDESS emotion codes."""
        cases = [
            ("01", "neutral"),
            ("02", "calm"),
            ("03", "happy"),
            ("04", "sad"),
            ("05", "angry"),
            ("06", "fearful"),
            ("07", "disgust"),
            ("08", "surprised"),
        ]
        for code, expected in cases:
            path = Path(f"RAVDESS/Actor_01/03-01-{code}-01-01-01-01.wav")
            _, emotion, _ = loader.parse_metadata(path)
            assert emotion == expected

    def test_parse_metadata_ravdess_invalid_format(self, loader):
        """parse_metadata raises MetadataParseError for wrong RAVDESS token count."""
        path = Path("RAVDESS/Actor_01/03-01-01.wav")
        with pytest.raises(MetadataParseError, match="Expected 7 hyphenated tokens"):
            loader.parse_metadata(path)

    def test_parse_metadata_ravdess_unknown_emotion_code(self, loader):
        """parse_metadata raises MetadataParseError for unknown RAVDESS emotion code."""
        path = Path("RAVDESS/Actor_01/03-01-99-01-01-01-01.wav")
        with pytest.raises(MetadataParseError, match="Unsupported RAVDESS emotion code"):
            loader.parse_metadata(path)

    def test_parse_metadata_cremad(self, loader):
        """parse_metadata correctly parses CREMA-D filename conventions."""
        path = Path("CREMA-D/1001_DFA_HAP_01.wav")
        dataset_name, emotion, speaker_id = loader.parse_metadata(path)
        assert dataset_name == "CREMA-D"
        assert emotion == "happy"
        assert speaker_id == "1001"

    def test_parse_metadata_cremad_all_emotions(self, loader):
        """parse_metadata handles all 6 CREMA-D emotion codes."""
        cases = [
            ("ANG", "angry"),
            ("DIS", "disgust"),
            ("FEA", "fearful"),
            ("HAP", "happy"),
            ("NEU", "neutral"),
            ("SAD", "sad"),
        ]
        for code, expected in cases:
            path = Path(f"CREMA-D/1001_DFA_{code}_01.wav")
            _, emotion, _ = loader.parse_metadata(path)
            assert emotion == expected

    def test_parse_metadata_cremad_invalid_format(self, loader):
        """parse_metadata raises MetadataParseError for wrong CREMA-D token count."""
        path = Path("CREMA-D/1001_HAP.wav")
        with pytest.raises(MetadataParseError, match="Expected 4 underscore-separated tokens"):
            loader.parse_metadata(path)

    def test_parse_metadata_unknown_dataset(self, loader):
        """parse_metadata raises DatasetDetectionError for unknown dataset paths."""
        path = Path("unknown_dataset/file.wav")
        with pytest.raises(DatasetDetectionError, match="Failed to identify dataset"):
            loader.parse_metadata(path)

    def test_load_sample(self, loader, mock_wav_dir):
        """load_sample creates a VoiceSample with correct metadata."""
        wav_path = mock_wav_dir / "RAVDESS" / "Actor_01" / "03-01-01-01-01-01-01.wav"
        sample = loader.load_sample(wav_path)
        assert isinstance(sample, VoiceSample)
        assert sample.dataset_name == "RAVDESS"
        assert sample.emotion == "neutral"
        assert sample.speaker_id == "01"
        assert sample.file_path.exists()
        # Audio is lazy loaded, not loaded yet
        assert sample.waveform is None
        assert sample.sample_rate is None

    def test_voice_sample_load_audio(self, loader, mock_wav_dir):
        """VoiceSample.load_audio loads waveform and caches it."""
        wav_path = mock_wav_dir / "CREMA-D" / "1001_DFA_HAP_01.wav"
        sample = loader.load_sample(wav_path)

        waveform, sr = sample.load_audio(sr=16000)
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert sr == 16000
        # Verify caching
        assert sample.waveform is waveform
        assert sample.sample_rate == sr

        # Second call should return cached data
        waveform2, sr2 = sample.load_audio(sr=16000)
        assert waveform2 is waveform

    def test_voice_sample_load_audio_missing_file(self, tmp_path):
        """VoiceSample.load_audio raises AudioLoadError for missing file."""
        wav_path = tmp_path / "nonexistent.wav"
        sample = VoiceSample(
            file_path=wav_path,
            dataset_name="test",
            emotion="neutral",
            speaker_id="0"
        )
        with pytest.raises(AudioLoadError, match="Failed to load"):
            sample.load_audio()

    def test_get_all_audio_files(self, loader, mock_wav_dir):
        """get_all_audio_files returns all WAV files recursively."""
        all_files = loader.get_all_audio_files(mock_wav_dir)
        assert len(all_files) == 6  # 3 RAVDESS + 3 CREMA-D
