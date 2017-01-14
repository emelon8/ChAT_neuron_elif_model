#ChAT model ELIF

import numpy as np
import matplotlib.pylab as pyl
import scipy as sp
from scipy import stats

def chat_elif(dt,C,gL,gLL,EL,ELL,Eb,slp,Vr,Vt,taub,taubh,tauw,gBmax,gnoise,filterfrequency,a,B,initv,IDC,pause_current,noise_start,pulse_duration,pulse_start,trials,shouldPlot):
    SR=(1/dt)*1000 #sample rate
    T=(pulse_start+pulse_duration+500)/1000 # total time to simulate (msec)
    t=np.arange(0,T*1000+dt,dt) # time array

    v=initv*np.ones((len(IDC),len(t)))
    Ib=np.zeros((len(IDC),len(t)))
    w=np.zeros((len(IDC),len(t)))
    Ie=pause_current*np.ones((len(IDC),len(t)))
    Inoise=np.zeros((len(IDC),len(t)))
    spikefrequency=np.zeros(len(IDC))
    spikeIDC=np.zeros(len(IDC))
    IBspike=np.zeros(len(IDC))
    spikeV=np.zeros(len(IDC))
    stdV=np.zeros(len(IDC))
    stdnoise=np.zeros(len(IDC))
    noiseV=np.zeros(len(IDC))
    #spikeINOUT=np.zeros((len(IDC),7))

    for k in range(len(IDC)):
        b=0.14+0.81/(1+np.exp((initv+22.46)/-8.08))
        bh=0.08+0.88/(1+np.exp((initv+60.23)/5.69)) # extracted from experiments
        spiketimes=[];

#        # noise generation
#        dt_ins=dt/1000
#        df=1/(T+dt_ins) # freq resolution
#        fidx=np.arange(1,len(t)/2,1) # it has to be N/2 pts, where N=len(t); Python makes a range from 1 to np.ceil(len(t)/2)-1
#        faxis=(fidx-1)*df
#        #make the phases
#        Rr=sp.randn(np.size(fidx)) # ~N(0,1) over [-1,1]
#        distribphases=np.exp(1j*np.pi*Rr) # on the unit circle
#        #make the amplitudes - filtered
#        # filterf=1./(1+faxis/filterfrequency); # see the PSD of an OU process,
#        filterf=sp.sqrt(1/((2*np.pi*filterfrequency)**2+(2*np.pi*faxis)**2))
#
#        fourierA=distribphases*filterf # representation in fourier domain
#        # make it conj-symmetric so the ifft is real
#        fourierB=fourierA.conj()[::-1]
#        nss=np.concatenate(([0],fourierA,fourierB))
#        Inoise[k,:]=np.fft.ifft(nss)
#        scaling=np.std(Inoise[k,:], ddof=1)
#        Inoise[k,:]=Inoise[k,:]/scaling
#        Inoise[k,:]=Inoise[k,:]*gnoise
#        # noise generation
        from Inoise import intrinsic_noise
        Inoise[k,:]=intrinsic_noise(dt,T,t,filterfrequency,gnoise)
        Inoise[k,0:noise_start/dt]=0
        Ie[k,pulse_start/dt:(pulse_start+pulse_duration)/dt]=IDC[k]

        for i in range(len(t)-1):
            ### slow K+ current
            binf=0.14+0.81/(1+np.exp((v[k,i]+22.46)/-8.08))
            bhinf=0.08+0.88/(1+np.exp((v[k,i]+60.23)/5.69))
            bdt=((binf-b)/taub)*dt
            bhdt=((bhinf-bh)/taubh)*dt

            b=b+bdt
            bh=bh+bhdt

            Ib[k,i]=b*bh*gBmax*(v[k,i]-Eb)

            ##voltage
            fv_i=(-gL*(v[k,i]-EL)+gL*slp*np.exp((v[k,i]-Vt)/slp)-w[k,i]-gLL*(v[k,i]-ELL)-Ib[k,i]+Ie[k,i]+Inoise[k,i])/C

            ##w (spike dependent adaptation current)
            dw=((-w[k,i]+a*(v[k,i]-EL))/tauw)*dt

            k1v=dt*fv_i

            v[k,i+1]=v[k,i]+k1v
            w[k,i+1]=w[k,i]+dw

            if v[k,i]>0 and pulse_start/dt<=i and i<(pulse_start+pulse_duration)/dt:
                v[k,i+1]=Vr
                v[k,i]=50
                w[k,i+1]=w[k,i]+B
                spiketimes=np.append(spiketimes,i*dt/1000)
                Ib[k,i]=b*bh*gBmax*(v[k,i]-Eb)
            elif v[k,i]>0:
                v[k,i+1]=Vr
                v[k,i]=50
                w[k,i+1]=w[k,i]+B
                Ib[k,i]=b*bh*gBmax*(v[k,i]-Eb)

        # spikefrequency=1./mean(diff(spiketimes));
        spikefrequency[k]=np.size(spiketimes)/(pulse_duration/1000)
        spikeIDC[k]=IDC[k]
        IBspike[k]=np.mean(Ib[k,(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt])
        spikeV[k]=np.mean(v[k,(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt])
        stdV[k]=np.std(v[k,(pulse_start+100)/dt:(pulse_start+pulse_duration)/dt], ddof=1)
        stdnoise[k]=np.std(v[k,noise_start/dt:pulse_start/dt-1], ddof=1)
        noiseV[k]=np.mean(v[k,noise_start/dt:pulse_start/dt-1])

        if shouldPlot:
            pyl.plot(t,v[k,:])

    #spikeINOUT[k,:]=[spikeV,spikeIDC,spikefrequency,IBspike,stdV,stdnoise,noiseV] #sums all the trials
    #f-I gains
    spikingpulses=np.arange(np.argmax(spikefrequency>0),np.argmax(spikefrequency==max(spikefrequency)),1)

    if np.size(spikingpulses)>1:
        linfit=stats.linregress(spikefrequency[spikingpulses],spikeIDC[spikingpulses])
        gains=linfit[0]
        rsquared=linfit[2]**2
        #f-V gains
        linfit_fv=stats.linregress(spikefrequency[spikingpulses],spikeV[spikingpulses])
        gains_fv=linfit_fv[0]
        rsquared_fv=linfit_fv[2]**2

        # find the mean range
        meanrange=np.ptp(spikeV[spikingpulses])
        minrange=min(spikeV[spikingpulses])
        maxrange=max(spikeV[spikingpulses])
    else:
        linfit=np.NaN
        gains=np.NaN
        rsquared=np.NaN
        linfit_fv=np.NaN
        gains_fv=np.NaN
        rsquared_fv=np.NaN
        meanrange=np.NaN
        minrange=np.NaN
        maxrange=np.NaN

    #find the resistance of the model neuron
    adjacent_subthreshold_pulses=spikeV[0:5]
    if np.size(adjacent_subthreshold_pulses)>1: #find if trial has multiple adjacent subthreshold pulses at the beginning
        Ri=np.mean(np.diff(adjacent_subthreshold_pulses))/(IDC[1]-IDC[0])
    else:
        Ri=np.NaN

    #find the average voltage standard deviation for this trial
    mean_trial_stdnoise=np.mean(stdnoise)
    std_trial_stdnoise=np.std(stdnoise, ddof=1)
    ste_trial_stdnoise=std_trial_stdnoise/sp.sqrt(np.size(stdnoise))
    mean_trial_noiseV=np.mean(noiseV)
    std_trial_noiseV=np.std(noiseV, ddof=1)
    ste_trial_noiseV=std_trial_noiseV/sp.sqrt(np.size(noiseV))

    ## plot membrane potential trace
    if shouldPlot:
        pyl.title('ChAT Neuron Voltage step')
        pyl.ylabel('membrane motential [mV]')
        pyl.xlabel('time [ms]')
        #ylim([0,2])
        pyl.show()
    return T,SR,t,Ri,spikeV,spikeIDC,spikefrequency,IBspike,stdV,stdnoise,noiseV,meanrange,minrange,maxrange,spiketimes,gains,rsquared,gains_fv,rsquared_fv,mean_trial_stdnoise,ste_trial_stdnoise,mean_trial_noiseV,ste_trial_noiseV #,gains,gains_fv
##mean Ri
#mean_Ri=nanmean(Ri)
#std_Ri=nanstd(Ri)
#ste_Ri=std_Ri/sqrt(size(~isnan(Ri)))
#
##mean noise
#mean_stdnoise=mean(mean_trial_stdnoise);
#std_stdnoise=std(mean_trial_stdnoise);
#ste_stdnoise=std_stdnoise/sqrt(size(mean_trial_stdnoise));
#mean_noiseV=mean(mean_trial_noiseV);
#std_noiseV=std(mean_trial_noiseV);
#ste_noiseV=std_noiseV/sqrt(size(mean_trial_noiseV));
#
##find the mean, max, and range of the f-V curve, a.k.a. the dynamic voltage window
#meanrange(meanrange==0)=NaN;
#minrange(meanrange==0)=NaN;
#maxrange(meanrange==0)=NaN;
#
#mean_min=nanmean(minrange);
#std_min=nanstd(minrange);
#ste_min=std_min/sqrt(sum(~isnan(minrange)));
#mean_max=nanmean(maxrange);
#std_max=nanstd(maxrange);
#ste_max=std_max/sqrt(sum(~isnan(maxrange)));
#
#mean_range=nanmean(meanrange);
#std_range=nanstd(meanrange);
#ste_range=std_range/sqrt(sum(~isnan(meanrange)));
#
##mean f-I gains
#mean_gains=mean(gains);
#std_gains=std(gains);
#ste_gains=std_gains/sqrt(size(~isnan(gains)));
#mean_rsquared=mean(rsquared);
#std_rsquared=std(rsquared);
#ste_rsquared=std_rsquared/sqrt(size(~isnan(rsquared)));
#
##mean f-V gains
#mean_gains_fv=mean(gains_fv);
#std_gains_fv=std(gains_fv);
#ste_gains_fv=std_gains_fv/sqrt(size(~isnan(gains_fv)));
#mean_rsquared_fv=mean(rsquared_fv);
#std_rsquared_fv=std(rsquared_fv);
#ste_rsquared_fv=std_rsquared_fv/sqrt(size(~isnan(rsquared_fv)));
#
#
##find the average spiking characteristics over all trials
#if trials>1:
#    mean_spikeINOUT=nanmean(spikeINOUT,1);mean_spikeINOUT=reshape(mean_spikeINOUT,[len(IDC) 7]);
#    std_spikeINOUT=nanstd(spikeINOUT,1);
#    ste_spikeINOUT=std_spikeINOUT/sqrt(size(~isnan(spikeINOUT(:,1,1))));ste_spikeINOUT=reshape(ste_spikeINOUT,[len(IDC) 7]);
#else:
#    mean_spikeINOUT=reshape(spikeINOUT,size(IDC),7);
#    ste_spikeINOUT=NaN;