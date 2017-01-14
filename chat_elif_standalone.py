import time
start_time=time.time()
import numpy as np
import matplotlib.pylab as pyl
import scipy as sp
from scipy import stats

shouldPlot = True

## Time and stimulus properties
IDC=255#arange(50,255,5) # current steps
pause_current=60 #200 for dep, 60 for hyp.
noise_start=1000 #[ms]
pulse_start=4000 #[ms]
pulse_duration=1000 #[ms]
trials=1 #number of times (trials) to rerun the model for the average

## LIF properties
dt=0.01 # [ms]
slp=10
C=81.9 # 81.9 capacitance [pF]
gL=1.3 # 3 base conductance for model, do not change
gLL=0 # 4 added leak conductance parameter
EL=-85 # reversal for EL
ELL=-60 # reversal for ELL
Eb=-93.1 # -85 reversal for Ib
# slp=10 # 10 slope factor or deltaT value for exponential term (0.1 is what papers commonly use: leak shifts f-I and doesn't do much to f-V)
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

#ChAT model ELIF

SR=(1/dt)*1000 #sample rate
T=(pulse_start+pulse_duration+500)/1000 # total time to simulate (msec)
t=np.arange(0,T*1000+dt,dt) # time array

v=initv*np.ones(len(t))
Ib=np.zeros(len(t))
w=np.zeros(len(t))
Ie=pause_current*np.ones(len(t))

b=0.14+0.81/(1+np.exp((initv+22.46)/-8.08))
bh=0.08+0.88/(1+np.exp((initv+60.23)/5.69)) # extracted from experiments
spiketimes=[];

# noise generation
dt_ins=dt/1000
df=1/(T+dt_ins) # freq resolution
fidx=np.arange(1,len(t)/2,1) # it has to be N/2 pts, where N=len(t); Python makes a range from 1 to np.ceil(len(t)/2)-1
faxis=(fidx-1)*df
#make the phases
Rr=sp.randn(np.size(fidx)) # ~N(0,1) over [-1,1]
distribphases=np.exp(1j*np.pi*Rr) # on the unit circle
#make the amplitudes - filtered
# filterf=1./(1+faxis/filterfrequency); # see the PSD of an OU process,
filterf=sp.sqrt(1/((2*np.pi*filterfrequency)**2+(2*np.pi*faxis)**2))

fourierA=distribphases*filterf # representation in fourier domain
# make it conj-symmetric so the ifft is real
fourierB=fourierA.conj()[::-1]
nss=np.concatenate(([0],fourierA,fourierB))
Inoise=np.fft.ifft(nss)
scaling=np.std(Inoise, ddof=1)
Inoise=Inoise/scaling
Inoise=Inoise*gnoise
# noise generation
Inoise[0:noise_start/dt]=0
Ie[pulse_start/dt:(pulse_start+pulse_duration)/dt]=IDC

for i in np.arange(0,len(t)-1):
    ### slow K+ current
    binf=0.14+0.81/(1+np.exp((v[i]+22.46)/-8.08))
    bhinf=0.08+0.88/(1+np.exp((v[i]+60.23)/5.69))
    bdt=((binf-b)/taub)*dt
    bhdt=((bhinf-bh)/taubh)*dt
    
    b=b+bdt
    bh=bh+bhdt
    
    Ib[i]=b*bh*gBmax*(v[i]-Eb)
    
    ##voltage
    fv_i=(-gL*(v[i]-EL)+gL*slp*np.exp((v[i]-Vt)/slp)-w[i]-gLL*(v[i]-ELL)-Ib[i]+Ie[i]+Inoise[i])/C
        
    ##w (spike dependent adaptation current)
    dw=((-w[i]+a*(v[i]-EL))/tauw)*dt
    
    k1v=dt*fv_i
    
    v[i+1]=v[i]+k1v
    w[i+1]=w[i]+dw
    
    if v[i]>0 and pulse_start/dt<=i and i<(pulse_start+pulse_duration)/dt:
        v[i+1]=Vr
        v[i]=50
        w[i+1]=w[i]+B
        spiketimes=np.append(spiketimes,i*dt/1000)
        Ib[i]=b*bh*gBmax*(v[i]-Eb)
    elif v[i]>0:
        v[i+1]=Vr
        v[i]=50
        w[i+1]=w[i]+B
        Ib[i]=b*bh*gBmax*(v[i]-Eb)

# spikefrequency=1./mean(diff(spiketimes));
spikefrequency=np.size(spiketimes)/(pulse_duration/1000)
spikeIDC=IDC
IBspike=np.mean(Ib[(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt])
spikeV=np.mean(v[(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt])
stdV=np.std(v[(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt], ddof=1)
stdnoise=np.std(v[noise_start/dt:pulse_start/dt-1], ddof=1)
noiseV=np.mean(v[noise_start/dt:pulse_start/dt-1])

spikeINOUT=[spikeV,spikeIDC,spikefrequency,IBspike,stdV,stdnoise,noiseV] #sums all the trials
#f-I gains
#    spikingpulses=np.arange(np.argmax(spikefrequency>0),np.argmax(spikefrequency==max(spikefrequency)),1)
#    
#    if np.size(spikingpulses)>1:
#        linfit=stats.linregress(spikefrequency[spikingpulses],spikeIDC[spikingpulses])
#        gains=linfit[0]
#        rsquared=linfit[2]**2
#        #f-V gains
#        linfit_fv=stats.linregress(spikefrequency[spikingpulses],spikeV[spikingpulses])
#        gains_fv=linfit_fv[0]
#        rsquared_fv=linfit_fv[2]**2
#        
#        # find the mean range
#        meanrange=np.ptp(spikeV[spikingpulses])
#        minrange=min(spikeV[spikingpulses])
#        maxrange=max(spikeV[spikingpulses])
#    else:
#        linfit=np.NaN
#        gains=np.NaN
#        rsquared=np.NaN
#        linfit_fv=np.NaN
#        gains_fv=np.NaN
#        rsquared_fv=np.NaN
#        meanrange=np.NaN
#        minrange=np.NaN
#        maxrange=np.NaN

#find the resistance of the model neuron
#    adjacent_subthreshold_pulses=spikeV[0:5]
#    if size(adjacent_subthreshold_pulses)>1: #find if trial has multiple adjacent subthreshold pulses at the beginning
#        Ri=mean(diff(adjacent_subthreshold_pulses))/(IDC(2)-IDC(1))
#    else:
#        Ri=NaN

#    #find the average voltage standard deviation for this trial
#    mean_trial_stdnoise=np.mean(stdnoise)
#    std_trial_stdnoise=np.std(stdnoise, ddof=1)
#    ste_trial_stdnoise=std_trial_stdnoise/sp.sqrt(np.size(stdnoise))
#    mean_trial_noiseV=np.mean(noiseV)
#    std_trial_noiseV=np.std(noiseV, ddof=1)
#    ste_trial_noiseV=std_trial_noiseV/sp.sqrt(np.size(noiseV))

## plot membrane potential trace  
if shouldPlot:
    pyl.plot(t,v)
    pyl.title('ChAT Neuron Voltage step')
    pyl.ylabel('Membrane Potential [mV]')
    pyl.xlabel('Time [ms]')
    #ylim([0,2])
    pyl.show()

total_time=time.time()-start_time #in seconds
print(total_time)
#maybe make the neuron a class, with different inputs being different methods?