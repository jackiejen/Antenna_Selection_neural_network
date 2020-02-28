def f(K,N,H):
    import numpy as np
    N=N         #发射天线数
    M=K		#接收、选择天线数
    times=1     #算法循环次数
    threshold=0.01;   #阈值 
    SNR=np.zeros(times)
    for num in range(times):
        H=np.matrix(H)
        HtH=np.dot(H,H.H)
        lam,eigv=np.linalg.eig(HtH)
        tempv=eigv[:,np.argsort(-np.real(lam))[0]].reshape(M,1)
        tempv=H.H*tempv
#        V=sorted(abs(tempv),reverse=True)
        I=np.argsort(-abs(tempv),axis=0).tolist()    
        tempv[I[K:H.shape[1]],:]=0
        x=tempv
        pre_x=0
        while np.linalg.norm(pre_x-x)>threshold:
            pre_x=x
            x=np.dot(H.H*H,x)
#            V=sorted(abs(x),reverse=True)
            I=np.argsort(-abs(x),axis=0).tolist()    
            x[I[K:H.shape[1]],:]=0
            x=x/np.linalg.norm(x)
        AntennaSele=I[0:K]                  
        H1=H[:,sum(AntennaSele,[])]
        lamd,_=np.linalg.eig(H1.H*H1)
        SNR[num]=np.max(np.real(lamd))
    return AntennaSele,SNR