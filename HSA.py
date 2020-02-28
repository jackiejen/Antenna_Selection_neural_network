def f(K,N,H):
    ########################################################
    import numpy as np     
    Beta=5
    N=N         #发射天线数
    M=K		#接收、选择天线数
    times=1     #算法循环次数
    threshold=0.01;   #阈值 
    SNR=np.zeros(times)
    ######################################################
    for num in range(times):
#        H=(np.random.normal(size=(M,N))\
#            +1j*np.random.normal(size=(M,N)))*np.sqrt(0.5)
        H=np.matrix(H)
        x=np.random.normal(size=(M,1))+1j*np.random.normal(size=(M,1))
        x=x/np.linalg.norm(x)
        pre_x=0
        while np.linalg.norm(pre_x-x)>threshold:
            pre_x=x
            T=np.dot(H.H,x)
            Htx=np.multiply(abs(T),abs(T))
            IniHtx=Htx
            Htx=np.sort(Htx,axis=0)      
            rho=(Htx[N-K]+Htx[N-K-1])/2
            temp=np.exp(Beta*(IniHtx-rho))
            alg=temp/(1+temp)
            diff_F=np.dot(2*H,np.multiply(alg,T))
            x=diff_F/np.linalg.norm(diff_F)
    
        SnrOrder=abs(np.dot(H.H,x))
        AntennaSele=np.argsort(-SnrOrder,axis=0)[0:K].tolist()                   
        H1=H[:,sum(AntennaSele,[])]
        lamd,_=np.linalg.eig(H1.H*H1)
        SNR[num]=np.max(np.real(lamd))
    
    return AntennaSele,SNR
    


    
        
        
     
        

    
    
        
    
    
    
