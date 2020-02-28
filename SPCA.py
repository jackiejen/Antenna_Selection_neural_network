def f(K,N,H):
    import numpy as np
    N=N         #发射天线数
#    M=K		#接收、选择天线数
    times=1     #算法循环次数
    threshold=0.01;   #阈值 
    SNR=np.zeros(times)
    for num in range(times):
        H=np.matrix(H)
        x=np.random.normal(size=(N,1))+1j*np.random.normal(size=(N,1))
        x=x/np.linalg.norm(x)
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