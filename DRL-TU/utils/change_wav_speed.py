import sox
import librosa
import os
import soundfile
import sys 

tfm = sox.Transformer()
audio_speed = sys.argv[1]
print("audio_speed is ",audio_speed)
tfm.speed(float(audio_speed))

utt2wav={i.split()[0]:i.split()[1] for i in open('./wav_18_dev.scp')}
for utt in utt2wav:
    signal,fs  = librosa.load(utt2wav[utt], sr=16000)
    y_out = tfm.build_array(input_array=signal, sample_rate_in=fs)
    save_path = os.path.join('/DATA1/vox2_speed/dev/speed0.9/',utt+'-speed09.wav')
    soundfile.write(save_path, y_out, fs)
