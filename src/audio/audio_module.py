from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)
DEFAULT_WHISPER_LANGUAGE = "en"


class AudioModule:
    def __init__(
        self,
        *,
        model_size: str = "tiny",
        language: str = DEFAULT_WHISPER_LANGUAGE,
        sample_rate: int = 16_000,
        block_duration_s: float = 0.10,
        listen_timeout_s: float = 2.0,
        max_record_s: float = 5.0,
        min_record_s: float = 0.30,
        silence_threshold: float = 0.006,
        trailing_silence_s: float = 0.80,
        device: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_duration_s))
        self.listen_timeout_s = listen_timeout_s
        self.max_record_s = max_record_s
        self.min_record_s = min_record_s
        self.silence_threshold = silence_threshold
        self.trailing_silence_s = trailing_silence_s
        self.device = device
        self._whisper_model = None
        self._whisper_cls = None
        self._sounddevice = None
        self._deps_loaded = False

    async def transcribe_audio(self) -> Optional[str]:
        self._ensure_dependencies_loaded()
        samples = await self._capture_voice_chunk()
        if samples is None:
            return None
        await self._ensure_whisper_model()
        return await asyncio.to_thread(self._transcribe_samples, samples)

    async def close(self) -> None:
        return

    def _ensure_dependencies_loaded(self) -> None:
        if self._deps_loaded:
            return
        try:
            import sounddevice as sounddevice
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Missing audio dependencies. Install requirements.txt") from exc

        self._sounddevice = sounddevice
        self._whisper_cls = WhisperModel
        self._deps_loaded = True

    async def _ensure_whisper_model(self) -> None:
        if self._whisper_model is not None:
            return
        assert self._whisper_cls is not None
        self._whisper_model = await asyncio.to_thread(
            self._whisper_cls,
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

    async def _capture_voice_chunk(self) -> Optional[np.ndarray]:
        assert self._sounddevice is not None
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        capture_started_at = loop.time()
        speech_started = False
        speech_started_at = 0.0
        last_speech_at = 0.0
        chunks: list[np.ndarray] = []

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.debug("Audio callback status: %s", status)
            mono = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
            loop.call_soon_threadsafe(queue.put_nowait, mono)

        with self._sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=callback,
            device=self.device,
        ):
            while True:
                chunk = await self._next_audio_chunk(queue, capture_started_at, speech_started, speech_started_at, last_speech_at)
                if chunk is None:
                    return None if not speech_started else self._validated_audio(chunks)

                has_voice = self._has_voice(chunk)
                if has_voice and not speech_started:
                    speech_started = True
                    speech_started_at = loop.time()
                    last_speech_at = speech_started_at
                if not speech_started:
                    continue
                chunks.append(chunk)
                if has_voice:
                    last_speech_at = loop.time()

    async def _next_audio_chunk(self, queue, capture_started_at, speech_started, speech_started_at, last_speech_at):
        now = asyncio.get_running_loop().time()
        if not speech_started and (now - capture_started_at) >= self.listen_timeout_s:
            return None
        if speech_started and (now - speech_started_at) >= self.max_record_s:
            return None
        if speech_started and (now - last_speech_at) >= self.trailing_silence_s:
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            return np.zeros((1,), dtype=np.float32)

    def _validated_audio(self, chunks: list[np.ndarray]) -> Optional[np.ndarray]:
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0)
        if audio.shape[0] / float(self.sample_rate) < self.min_record_s:
            return None
        return audio.astype(np.float32, copy=False)

    def _has_voice(self, chunk: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)) >= self.silence_threshold

    def _transcribe_samples(self, samples: np.ndarray) -> Optional[str]:
        assert self._whisper_model is not None
        segments, _ = self._whisper_model.transcribe(
            samples,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        return text or None


async def transcribe_audio(audio_module: AudioModule) -> Optional[str]:
    return await audio_module.transcribe_audio()
