from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset)
    for k, v in inputs.items():
        inputs[k] = v
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

text = "♪ A toalha de corpo não é feita de corpo, ela é feita de pano. A toalha de corpo é uma toalha de pano que enxuga o corpo. A toalha de rosto não é feita de rosto, ela é feita de pano. A toalha de rosto é uma toalha de pano que enxuga o rosto ♪"

generate_audio(
    text=text,
    preset="v2/pt_speaker_9",
    output="basic.wav"
)