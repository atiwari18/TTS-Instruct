import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

#Load Models
HF_REPO ="LiquidAI/LFM2.5-Audio-1.5B"
processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

#Set up chat
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio")
chat.end_turn()

#Start a new turn
chat.new_turn("user")
chat.add_text("My business specialized in walking dogs, can you give me some names related to that?")
chat.end_turn()

#Generate turn text and audio tokens
chat.new_turn("assistant")

#Generate turn text and audio tokens.
audio_out: list[torch.Tensor] = []
for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)

    else:
        audio_out.append(t)

#Detokenize audio and removing the last "end of audio" token
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
torchaudio.save("answer1.mp3", waveform.cpu(), 24_000)