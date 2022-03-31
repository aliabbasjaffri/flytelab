import bentoml
import torchaudio
from io import BytesIO
from pydub import AudioSegment
from preprocess import chop_audio
from bentoml.io import File, JSON
from torchaudio.datasets.gtzan import gtzan_genres

max_size = 28 * 1000  # max size of audio we take (for 28 sec)
segment_size = 2 * 1000  # segment size(for 2 sec)
artifact_name = "genre_classification_model"
genre_classifier_runner = bentoml.pytorch.load_runner(tag=artifact_name)
svc = bentoml.Service(name=artifact_name, runners=[genre_classifier_runner])


@svc.api(input=File(), output=JSON())
def classify(input_music_file) -> str:
    audio_seg = AudioSegment.from_wav(input_music_file)
    audio_length = len(audio_seg)

    print(audio_length)

    if audio_length > segment_size:
        preprocessed_audio = chop_audio(audio_file=input_music_file, segment_size=segment_size, max_size=max_size)[0]
        preprocessed_audio = preprocessed_audio.export(input_music_file, format="wav")
    else:
        print("came here...")
        preprocessed_audio = input_music_file

    x = torchaudio.load(preprocessed_audio)[0]
    prediction_probabilities = genre_classifier_runner.run(x)
    print(prediction_probabilities.max())
    value = gtzan_genres[(prediction_probabilities.max()).nonzero(as_tuple=True)[0]]
    print(value)
    return value
