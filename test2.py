import EigHSA,HSA,EigSA,SA,EigSPCA,SPCA,AA
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
def TimeSNR(function,K,N,H):
    start=time.time()
    antenna,snr=function(K,N,H)
    end=time.time()
    Time=end-start
    return Time,snr

K=2
M=K
count=100
TransN=[4,8,12,16]
Arg=['AA','NN']#['EigHSA','HSA','EigSA','SA','EigSPCA','SPCA','AA','NN']
ArgNum=len(Arg)
averSNR=np.zeros([ArgNum,len(TransN)])
averTime=np.zeros([ArgNum,len(TransN)])
divTime=np.zeros([ArgNum-1,len(TransN)])
for N in TransN:
    snr=np.zeros([ArgNum,count])
    runtime=np.zeros([ArgNum,count])
    print (N)
     #-------------------------------------------------------------------------#   
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    mPath="model2/model%s_%s"%(N,M)
    xcol=M*N
    ycol=N
    hiddenSize1=64
    hiddenSize2=64
#    hiddenSize3=2*N
    weights = {
        'h1': tf.Variable(tf.truncated_normal([xcol, hiddenSize1],stddev=np.sqrt(2.0/xcol))),
        'h2': tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize2],stddev=np.sqrt(2.0/hiddenSize1))),
#        'h3': tf.Variable(tf.truncated_normal([hiddenSize2, hiddenSize3],stddev=np.sqrt(2.0/hiddenSize2))),  
        'out': tf.Variable(tf.truncated_normal([hiddenSize2, ycol],stddev=np.sqrt(2.0/hiddenSize2)))
    }
    biases = {
        'b1': tf.Variable(tf.ones([hiddenSize1])*0.1),
        'b2': tf.Variable(tf.ones([hiddenSize2])*0.1),
#        'b3': tf.Variable(tf.ones([hiddenSize3])*0.1),
        'out': tf.Variable(tf.ones([ycol])*0.1)
    }
    def BNnet(x,decay = 0.999,epsilon = 1e-5):
        mean, var = tf.nn.moments(x,[0])
        scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))     
        ema = tf.train.ExponentialMovingAverage(decay)
        def mean_var():
            ema_op=ema.apply([mean,var]) 
            with tf.control_dependencies([ema_op]):
                return tf.identity(mean), tf.identity(var)      
        mean,var=tf.cond(isTrain,mean_var,lambda:(ema.average(mean),ema.average(var)))      
        return tf.nn.batch_normalization(x,mean,var,beta,scale,epsilon) 
    def GDNN(x, weights, biases):
        L1 = tf.nn.relu(tf.matmul(x,weights['h1'])+biases['b1'])
 #       L1 = BNnet(L1)
        
        L2 = tf.nn.relu(tf.matmul(L1,weights['h2'])+biases['b2'])
 #       L2 = BNnet(L2)
        
#        L3 = tf.nn.relu(tf.matmul(L2,weights['h3'])+biases['b3'])
#        L3 = BNnet(L3)
        yout = tf.matmul(L2,weights['out'])+biases['out']
 #       yout=tf.nn.relu6(tf.matmul(L2,weights['out'])+biases['out'])/6
        return yout
    x=tf.placeholder(tf.float32, [None,xcol])
    isTrain=tf.placeholder(tf.bool)

    yout=GDNN(x, weights, biases)

    _,NNAS=tf.nn.top_k(yout, k=K)
    NNAS,_=tf.nn.top_k(NNAS,k=K)
    saver = tf.train.Saver()
#    print("test predict from the model ")
    saver.restore(sess, mPath+"/model.ckpt")  
        #-------------------------------------------------------------------------#
    for i in range(count):
        H=(np.random.normal(size=(M,N))+1j*np.random.normal(size=(M,N)))*np.sqrt(0.5)
        Ht=np.matrix(abs(H))	
        testH=sum(Ht.tolist(),[])
        
#        runtime[Arg.index('EigHSA'),i],snr[Arg.index('EigHSA'),i]=TimeSNR(EigHSA.f,K,N,H)
        # runtime[Arg.index('HSA'),i],snr[Arg.index('HSA'),i]=TimeSNR(HSA.f,K,N,H)
        # runtime[Arg.index('EigSA'),i],snr[Arg.index('EigSA'),i]=TimeSNR(EigSA.f,K,N,H)
        # runtime[Arg.index('SA'),i],snr[Arg.index('SA'),i]=TimeSNR(SA.f,K,N,H)
        # runtime[Arg.index('EigSPCA'),i],snr[Arg.index('EigSPCA'),i]=TimeSNR(EigSPCA.f,K,N,H)
        # runtime[Arg.index('SPCA'),i],snr[Arg.index('SPCA'),i]=TimeSNR(SPCA.f,K,N,H)
        runtime[Arg.index('AA'),i],snr[Arg.index('AA'),i]=TimeSNR(AA.f,K,N,H)
        if i%100==0 and i!=0:
            print(i)

        tic=time.time()
        NNAS0=sess.run(NNAS,feed_dict={
                          x:np.matrix(testH),
                          isTrain:False,
                          }) 
        H1=np.matrix(H[:,sum(NNAS0.tolist(),[])])
        eigv,_=np.linalg.eig(H1.H*H1)
        snr[Arg.index('NN'),i]=np.max(np.real(eigv))
        toc=time.time()
        runtime[Arg.index('NN'),i]=toc-tic
    s=np.sum(snr,axis=1)/count
    t=np.sum(runtime,axis=1)/count
    ite=TransN.index(N)
    averSNR[:,ite]=s
    averTime[:,ite]=t
    divTime[:,ite]=t[0:len(t)-1]/(t[-1]*np.ones(len(t)-1))
X=TransN
#plt.figure(figsize=(14, 8))  
plt.figure(1) 
#ax1=plt.subplot(3,1,1)
#plt.plot(X,averSNR[Arg.index('EigHSA'),:],'kd-',label="HSA")
# plt.plot(X,averSNR[Arg.index('HSA'),:],'y*-',label="HSA")
# plt.plot(X,averSNR[Arg.index('EigSA'),:],'g+-',label="EigSA")
# plt.plot(X,averSNR[Arg.index('SA'),:],'bs-',label="SA")
# plt.plot(X,averSNR[Arg.index('EigSPCA'),:],'mo-',label="EigSPCA")
# plt.plot(X,averSNR[Arg.index('SPCA'),:],'c<-',label="SPCA")
plt.plot(X,averSNR[Arg.index('AA'),:],'c<-',label="Algorithm in [1]")
plt.plot(X,averSNR[Arg.index('NN'),:],'r^-',label="Neural Network")
plt.legend(loc='upper left')
plt.ylabel('Average SNR')
plt.xlabel('Number of transmitter antennas N')

###-----------------------------------------------------------------##
#plt=plt.subplot(3,1,2)
plt.figure(2)
#plt.plot(X,averTime[Arg.index('EigHSA'),:],'kd-',label="HSA")
# plt.plot(X,averTime[Arg.index('HSA'),:],'y*-',label="HSA")
# plt.plot(X,averTime[Arg.index('EigSA'),:],'g+-',label="EigSA")
# plt.plot(X,averTime[Arg.index('SA'),:],'bs-',label="SA")
# plt.plot(X,averTime[Arg.index('EigSPCA'),:],'mo-',label="EigSPCA")
# plt.plot(X,averTime[Arg.index('SPCA'),:],'c<-',label="SPCA")
plt.plot(X,averTime[Arg.index('AA'),:],'c<-',label="Algorithm in [1]")
plt.plot(X,averTime[Arg.index('NN'),:],'r^-',label="Neural Network")
plt.legend(loc='upper left')
plt.ylabel('Average Runtime')
plt.xlabel('Number of transmitter antennas N')

##-----------------------------------------------------------------##
#ax3=plt.subplot(3,1,3)
# plt.figure(3)
# plt.plot(X,divTime[Arg.index('EigHSA'),:],'kd-',label="EigHSA")
# plt.plot(X,divTime[Arg.index('HSA'),:],'y*-',label="HSA")
# plt.plot(X,divTime[Arg.index('EigSA'),:],'g+-',label="EigSA")
# plt.plot(X,divTime[Arg.index('SA'),:],'bs-',label="SA")
# plt.plot(X,divTime[Arg.index('EigSPCA'),:],'mo-',label="EigSPCA")
# plt.plot(X,divTime[Arg.index('SPCA'),:],'c<-',label="SPCA")
# plt.plot(X,divTime[Arg.index('AA'),:],'c<-',label="AA")
# plt.legend(loc='upper left')
# plt.ylabel('Div Runtime')
# plt.xlabel('Number of transmitter antennas N')
plt.show()

# print(averSNR)
##print(averTime)    
#    
