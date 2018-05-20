import librosa
import glob
import stempeg

l = glob.glob('./*')
for i in range(len(l)):
     S, rate = stempeg.read_stems(l[i], stem_id=[0,1,2,3,4])
     A = np.sum(S, axis=0)
     librosa.output.write_wav(str(i)+'.wav', S[:,0],sr=44100)
