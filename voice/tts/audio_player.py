"""Audio playback for generated speech."""

from typing import Optional, Union

import sounddevice as sd

from voice.text_to_speech import SpeechAudio


class SoundDeviceAudioPlayer:
    """Play generated PCM audio through sounddevice."""

    def __init__(self, device: Optional[Union[int, str]] = None):
        self.device = device

    def play(self, audio: SpeechAudio) -> None:
        """Play audio and block until playback completes."""
        if audio.samples.size:
            sd.play(
                audio.samples,
                samplerate=audio.sample_rate,
                device=self.device,
                blocking=True,
            )

    def stop(self) -> None:
        """Stop active playback."""
        sd.stop(device=self.device)
