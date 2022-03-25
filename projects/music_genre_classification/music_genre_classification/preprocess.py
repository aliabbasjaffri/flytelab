import os
from pydub import AudioSegment
from utils import generate_directory
from torchaudio.datasets.gtzan import gtzan_genres

preprocessed_data_dir = "data/"
genre_dir = "genres/"
max_sec = 28 * 1000  # max size of audio we take (for 28 sec)
segment = 2 * 1000  # segment size(for 2 sec)


def export_to_dir(class_name, data_dir, chopped_audio, counter):
    for audio_data in chopped_audio:
        file_name = class_name + "_" + str(counter) + ".wav"
        export_path = os.path.join(data_dir, file_name)
        audio_data.export(export_path, format="wav")
        int(counter)
        counter += 1
    return counter


def chop_audio(
    audio_file: str, segment_size: int = segment, max_size: int = max_sec
):  # All file are not even so we cut it in same size using max_size and provide segment size using segment
    audio_data = AudioSegment.from_wav(file=audio_file)
    cut_audio_data = audio_data[:max_size]
    chopped_audio = [x for x in cut_audio_data[::segment_size]]
    return chopped_audio


def preprocess_data(preprocessed_data_dir: str = "data/") -> None:
    generate_directory(preprocessed_data_dir)
    _counter = 0
    if len(os.listdir(preprocessed_data_dir)) == 0:
        for genre in gtzan_genres:
            path = os.path.join(genre_dir, genre)
            audio_files = os.listdir(path)
            for audio_file in audio_files:
                chopped_audio = chop_audio(audio_file=os.path.join(path, audio_file))
                _counter = export_to_dir(
                    genre, preprocessed_data_dir, chopped_audio, _counter
                )
                print(audio_file)
    else:
        print("data already preprocessed")


if __name__ == "__main__":
    preprocess_data()
