import time
start_time=time.time()
import numpy as np
import chat_model as cm

shouldPlot = True

## Time and stimulus properties
IDC=np.arange(50,255,5) # current steps
pause_current=200 #200 for dep, 60 for hyp.
noise_start=3000 #[ms]
pulse_start=6000 #[ms]
pulse_duration=1000 #[ms]
trials=1 #number of times (trials) to rerun the model for the average

## LIF properties
dt=0.01 # [ms]
C=81.9 # 81.9 capacitance [pF]
gL=1.3 # 3 base conductance for model, do not change
gLL=0 # 4 added leak conductance parameter
EL=-85 # reversal for EL
ELL=-60 # reversal for ELL
Eb=-93.1 # -85 reversal for Ib
slp=10 # 10 slope factor or deltaT value for exponential term (0.1 is what papers commonly use: leak shifts f-I and doesn't do much to f-V)
Vr=-65 # -65 reset value
Vt=-59.5 # threshhold value
taub=152.7 # 180
taubh=11100 # 5000
tauw=125 # w time constant
gBmax=30.1 # 34.1 slow, potassium current maximum conductance
gnoise=11 # 7 for leak, not really a conductance, just a multiplier
filterfrequency=100 # cutoff frequency [Hz]
a=0.1 # w parameters (conductance)
B=2.5 # w parameters (increment per spike)
initv=-75

T,SR,t,Ri,spikeV,spikeIDC,spikefrequency,IBspike,stdV,stdnoise,noiseV,meanrange,minrange,maxrange,spiketimes,gains,rsquared,gains_fv,rsquared_fv,mean_trial_stdnoise,ste_trial_stdnoise,mean_trial_noiseV,ste_trial_noiseV=cm.chat_elif(dt,C,gL,gLL,EL,ELL,Eb,slp,Vr,Vt,taub,taubh,tauw,gBmax,gnoise,filterfrequency,a,B,initv,IDC,pause_current,noise_start,pulse_duration,pulse_start,trials,shouldPlot)

#np.savez("C:\\Users\\eric\\Dropbox\\Documents\\School\\aim3\\chat_model\\de",
#         T=T,SR=SR,t=t,Ri=Ri,spikeV=spikeV,spikeIDC=spikeIDC,spikefrequency=spikefrequency,
#         IBspike=IBspike,stdV=stdV,stdnoise=stdnoise,noiseV=noiseV,meanrange=meanrange,
#         minrange=minrange,maxrange=maxrange,spiketimes=spiketimes,gains=gains,
#         rsquared=rsquared,gains_fv=gains_fv,rsquared_fv=rsquared_fv,
#         mean_trial_stdnoise=mean_trial_stdnoise,ste_trial_stdnoise=ste_trial_stdnoise,
#         mean_trial_noiseV=mean_trial_noiseV,ste_trial_noiseV=ste_trial_noiseV,
#         dt=dt,C=C,gL=gL,gLL=gLL,EL=EL,ELL=ELL,Eb=Eb,slp=slp,Vr=Vr,Vt=Vt,taub=taub,
#         taubh=taubh,tauw=tauw,gBmax=gBmax,gnoise=gnoise,filterfrequency=filterfrequency,
#         a=a,B=B,initv=initv,IDC=IDC,pause_current=pause_current,noise_start=noise_start,
#         pulse_duration=pulse_duration,pulse_start=pulse_start,trials=trials,shouldPlot=shouldPlot)

total_time=time.time()-start_time #in seconds
#maybe make the neuron a class, with different inputs being different methods?