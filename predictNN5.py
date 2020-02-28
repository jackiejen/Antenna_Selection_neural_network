import tensorflow as tf
import numpy as np
import scipy.io as sio
import time

mPath="D:/NN/64x4/model"
###################|'data'|scope|####################################
DataPath='data10w/data4'
dataH_test=sio.loadmat(DataPath+'/dataH_test.mat')
testH=sio.loadmat(DataPath+'/testH.mat')
testAS=sio.loadmat(DataPath+'/testAS.mat')

H=dataH_test['dataH_test']
xtest=testH['testH']
ytest=testAS['testAS']
####################|'var'|scope|####################################
xcol=xtest.shape[1]
ycol=ytest.shape[1]
numtest=xtest.shape[0]

def testTime():
    snr1=0
    snr2=0
    consum=0
    K=4
    hiddenSize1=256
    hiddenSize2=256
    hiddenSize3=128
    hiddenSize4=128
    hiddenSize5=64
    ####################|'NN'|scope|############################################
    weights = {
        'h1': tf.Variable(tf.truncated_normal([xcol, hiddenSize1],stddev=np.sqrt(2.0/xcol))),
        'h2': tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize2],stddev=np.sqrt(2.0/hiddenSize1))),
        'h3': tf.Variable(tf.truncated_normal([hiddenSize2, hiddenSize3],stddev=np.sqrt(2.0/hiddenSize2))),
        'h4': tf.Variable(tf.truncated_normal([hiddenSize3, hiddenSize4],stddev=np.sqrt(2.0/hiddenSize3))),    
        'h5': tf.Variable(tf.truncated_normal([hiddenSize4, hiddenSize5],stddev=np.sqrt(2.0/hiddenSize4))),  
        'out': tf.Variable(tf.truncated_normal([hiddenSize5, ycol],stddev=np.sqrt(2.0/hiddenSize5)))
        }
    biases = {
        'b1': tf.Variable(tf.ones([hiddenSize1])*0.1),
        'b2': tf.Variable(tf.ones([hiddenSize2])*0.1),
        'b3': tf.Variable(tf.ones([hiddenSize3])*0.1),
        'b4': tf.Variable(tf.ones([hiddenSize4])*0.1),
        'b5': tf.Variable(tf.ones([hiddenSize5])*0.1),
        'out': tf.Variable(tf.ones([ycol])*0.1)
    }
    ###################|'Function'|scope|###################################
    def BNnet(x,decay = 0.99,epsilon = 1e-5):
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
        L1 = BNnet(L1)
        
        L2 = tf.nn.relu(tf.matmul(L1,weights['h2'])+biases['b2'])
        L2 = BNnet(L2)
        
        L3 = tf.nn.relu(tf.matmul(L2,weights['h3'])+biases['b3'])
        L3 = BNnet(L3)
        
        L4 = tf.nn.relu(tf.matmul(L3,weights['h4'])+biases['b4'])
        L4 = BNnet(L4)

        L5 = tf.nn.relu(tf.matmul(L4,weights['h5'])+biases['b5'])
        L5 = BNnet(L5)

        yout=tf.nn.relu6(tf.matmul(L5,weights['out'])+biases['out'])/6
        return yout

    ###################|'DNN'|scope|######################################
    x=tf.placeholder(tf.float32, [None,xcol])
    y=tf.placeholder(tf.float32, [None,ycol])
    isTrain=tf.placeholder(tf.bool)

    yout=GDNN(x, weights, biases)
#    loss = tf.reduce_mean(tf.square(tf.subtract(yout,y)))

    ###################|'predict'|scope|###################################
    _,NNAS=tf.nn.top_k(yout, k=K)
    _,ytestAS=tf.nn.top_k(y, k=K)
    NNAS,_=tf.nn.top_k(NNAS,k=K)
    ytestAS,_=tf.nn.top_k(ytestAS,k=K)
#    test_prediction = tf.equal(NNAS,ytestAS)
#    testAccuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))
    ###############################################################################
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        ckpt = tf.train.get_checkpoint_state(mPath)
        if ckpt and ckpt.model_checkpoint_path:
            print("test predict from the model ")
            saver.restore(sess, ckpt.model_checkpoint_path)  
        for i in range(numtest):
            tic=time.time()  
            NNAS0,ytestAS0=sess.run([NNAS,ytestAS],feed_dict={
                                  x:np.matrix(xtest[i,:]),
                                  y:np.matrix(ytest[i,:]),
                                  isTrain:False,
                                  }) 
            
            # test_loss,testAcc = sess.run([loss,testAccuracy],feed_dict = {x:xtest,y:ytest,isTrain:False,drop:1.0})
            # print ('###########################')
            # print ('loss:',test_loss)
            # print ('testAccuracy:',testAcc)
            # print ('###########################')
            toc=time.time()
            consum+=(toc-tic)
    
            H1=np.matrix(H[:,NNAS0[0],i])
            H2=np.matrix(H[:,ytestAS0[0],i])
            eigv1,_=np.linalg.eig(H1.H*H1)
            eigv2,_=np.linalg.eig(H2.H*H2)
            snr1=snr1+np.max(np.real(eigv1))
            snr2=snr2+np.max(np.real(eigv2))

        print ('---------------------------')
        print ('snr1:',snr1/numtest)
        print ('snr2:',snr2/numtest)
        print ('snrAcc:',snr1/snr2)
        print ('---------------------------')
        print ('time:',consum/numtest)
if __name__=='__main__':
    # tic=time.time()
    testTime()
    # toc=time.time()
    # print ('time:',(toc-tic)/100)
#    sio.savemat('data/NNAS.mat',{'NNAS':NNAS})