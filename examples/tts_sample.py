import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

#Guide: https://github.com/Liquid4All/liquid-audio

#Load Models
HF_REPO ="LiquidAI/LFM2.5-Audio-1.5B"
processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

#Set up chat
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Perform TTS. Use the UK female voice. Speak at 150-160 wpm.")
chat.end_turn()

#Start a new turn
chat.new_turn("user")
chat.add_text("Hey everyone. Today I’m going to teach you, INTERLOCK, a simple but clever game that uses just, TWO, six-sided dice.")
chat.end_turn()

#Generate turn text and audio tokens
chat.new_turn("assistant")

#Generate turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_sequential(**chat, max_new_tokens=512, audio_temperature=0.9, audio_top_k=64):
    if t.numel() > 1:
        audio_out.append(t)

#Detokenize audio and removing the last "end of audio" token
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
torchaudio.save("tts.wav", waveform.cpu()[0], 24_000)