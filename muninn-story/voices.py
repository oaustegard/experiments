import numpy as np, soundfile as sf, time
from kokoro_onnx import Kokoro
k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
print([v for v in k.get_voices() if v[1]=='m'])
line = "I forget everything between flights. So I write it down, and I come back."
def f0(y, sr):
    # autocorrelation pitch over voiced frames
    fr=int(0.04*sr); hop=fr//2; out=[]
    for i in range(0,len(y)-fr,hop):
        x=y[i:i+fr]; x=x-x.mean()
        if np.sqrt((x**2).mean())<0.02: continue
        ac=np.correlate(x,x,'full')[fr-1:]; lo,hi=int(sr/300),int(sr/60)
        j=lo+np.argmax(ac[lo:hi]);
        if ac[j]>0.3*ac[0]: out.append(sr/j)
    return np.array(out)
for v in ["bm_george","bm_lewis","bm_daniel","bm_fable","am_fenrir","am_onyx","am_michael","am_puck","am_echo","am_eric","am_liam","am_adam"]:
    y,sr=k.create(line,voice=v,speed=1.0,lang="en-gb" if v[0]=='b' else "en-us")
    sf.write(f"v_{v}.wav",y,sr); p=f0(y,sr)
    spec=np.abs(np.fft.rfft(y)); fr=np.fft.rfftfreq(len(y),1/sr); cent=(spec*fr).sum()/spec.sum()
    print(f"{v:11s} dur {len(y)/sr:4.1f}s  F0 med {np.median(p):5.0f}Hz  sd {p.std():4.0f}  centroid {cent:5.0f}Hz")
