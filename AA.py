def f(K,N,H):
    import numpy as np
    import cmath
#    K=2
#    M=2
#    N=5
#    H=(np.random.normal(size=(M,N))+1j*np.random.normal(size=(M,N)))*np.sqrt(0.5)
    S=[]
    snr_max=0
    [m,n]=H.shape
    for ite1 in range(n):
        for ite2 in range(ite1+1,n):
            for ite3 in range(ite2+1,n):
                dtemp=np.matrix(np.concatenate(([H[:,ite2]],[H[:,ite3]])).T)
                d=dtemp.I*np.matrix(H[:,ite1]).T
                D=(1-d.H*d)*np.matrix(2*abs(d[0,0]*d[1,0])).I
                if abs(D)<=1:
                    for sgn in range(-1,2,2):
                        fsi=sgn*cmath.acos(D)-np.imag(cmath.log(d[0,0]*np.conj(d[1,0])))
                        lam=np.imag(cmath.log([1,np.exp(-cmath.sqrt(-1) * fsi)]*d))
                        miu= lam+fsi
                        A=[np.exp(cmath.sqrt(-1)*lam) * H[:,ite1] - H[:,ite2],np.exp(cmath.sqrt(-1)* miu)* H[:,ite1] - H[:,ite3]]
                        A=np.conj(np.matrix(A))
                        _, _, eigd = np.linalg.svd(A[0,:])
                        c=eigd[-1,:].H
                        u=abs(np.matrix(H).H*c)
    #                        V=sorted(u,reverse=True)
                        I=sum(np.argsort(-u,axis=0).tolist(),[])  
                        lengthI=K
                        temp=[ite1,ite2,ite3]
                        temp2=I[K-1:K+2]
                        same_number=0
                        for q in range(3):
                            for r in range(3):
                                if temp[q]==temp2[r]:
                                    same_number+=1
                        if same_number==3:
                            lengthI=K+2
                        else:
                            temp2=I[K-2:K+1]
                            same_number=0
                            for q in range(3):
                                for r in range(3):
                                    if temp[q]==temp2[r]:
                                        same_number+=1
                            if same_number==3:
                                lengthI=K+1
                        if lengthI==K:
                            S=sum([S,I[0:K]],[])
                        elif lengthI==K+1:
                            I=I[0:lengthI]
                            if ite1 in I:
                                del I[I.index(ite1)]
                            if ite2 in I:
                                del I[I.index(ite2)]
                            if ite3 in I:
                                del I[I.index(ite3)]
                            H1=np.matrix(H[:,I+[ite1,ite2]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite1,ite2]
                            H1=np.matrix(H[:,I+[ite1,ite3]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite1,ite3]
                            H1=np.matrix(H[:,I+[ite2,ite3]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite2,ite3]
                        else:
                            I=I[0:lengthI]
                            if ite1 in I:
                                del I[I.index(ite1)]
                            if ite2 in I:
                                del I[I.index(ite2)]
                            if ite3 in I:
                                del I[I.index(ite3)]
                            H1=np.matrix(H[:,I+[ite1]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite1]
                            H1=np.matrix(H[:,I+[ite2]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite2]
                            H1=np.matrix(H[:,I+[ite3]])
                            eigv,_=np.linalg.eig(H1.H*H1)
                            snr=np.max(np.real(eigv))
                            if snr_max<snr:
                                snr_max=snr
                                AntennaSele=I+[ite3]
    
    for ite4 in range(n):
        for ite5 in range (ite4+1,n):
            _, _, eigd = np.linalg.svd(np.matrix(H[:,ite4]).H.T-np.matrix(H[:,ite5]).H.T)
            c=eigd[-1,:].H
            u=abs(np.matrix(H).H*c[:,0])
    #            V=sorted(u,reverse=True)
            I=sum(np.argsort(-u,axis=0).tolist(),[])  
            lengthI=K
            temp=[ite4,ite5]
            temp2=I[K-1:K+1]
            same_number=0
            for q in range(2):
                for r in range(2):
                    if temp[q]==temp2[r]:
                        same_number+=1
            if same_number==2:
                lengthI=K+1
            if lengthI==K:
                I=I[0:lengthI]
                H1=np.matrix(H[:,I])
                eigv,_=np.linalg.eig(H1.H*H1)
                snr=np.max(np.real(eigv))
                if snr_max<snr:
                    snr_max=snr
                    AntennaSele=I
            else:
                I=I[0:lengthI]
                if ite4 in I:
                    del I[I.index(ite4)]
                if ite5 in I:
                    del I[I.index(ite5)]
                H1=np.matrix(H[:,np.concatenate((I,[ite4]))])
                eigv,_=np.linalg.eig(H1.H*H1)
                snr=np.max(np.real(eigv))
                if snr_max<snr:
                    snr_max=snr
                    AntennaSele=np.concatenate((I,[ite4]))
                H1=np.matrix(H[:,np.concatenate((I,[ite5]))])
                eigv,_=np.linalg.eig(H1.H*H1)
                snr=np.max(np.real(eigv))
                if snr_max<snr:
                    snr_max=snr
                    AntennaSele=np.concatenate((I,[ite5]))
    return AntennaSele,snr_max
            
        
