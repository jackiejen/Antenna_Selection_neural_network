import tensorflow as tf
import numpy as np
import scipy.io as sio
mPath="D:/NN/64x4/model3"
K=4
batch=512
maxIter=50000

hiddenSize1=256
hiddenSize2=128
hiddenSize3=64

def Train():
    #################|'data'|scope|########################################
    DataPath='data10w/data1'
    dataH_test=sio.loadmat(DataPath+'/dataH_test.mat')
    trainH=sio.loadmat(DataPath+'/trainH.mat')
    trainAS=sio.loadmat(DataPath+'/trainAS.mat')
    testH=sio.loadmat(DataPath+'/testH.mat')
    testAS=sio.loadmat(DataPath+'/testAS.mat')
    
    H=dataH_test['dataH_test']
    xtrain=trainH['trainH']
    ytrain=trainAS['trainAS']
    xtest=testH['testH']
    ytest=testAS['testAS']
    ###################|'var'|scope|######################################
    xcol=xtrain.shape[1]
    ycol=ytrain.shape[1]
    numData=xtrain.shape[0]
    numtest=xtest.shape[0]
    snr1=0
    snr2=0
    weights = {
        'h01': tf.Variable(tf.truncated_normal([xcol, hiddenSize1],stddev=np.sqrt(2.0/xcol))),
        'h02': tf.Variable(tf.truncated_normal([xcol, hiddenSize2], stddev=np.sqrt(2.0 / xcol))),
        'h03': tf.Variable(tf.truncated_normal([xcol, hiddenSize3], stddev=np.sqrt(2.0 / xcol))),
        'h12': tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize2],stddev=np.sqrt(2.0/hiddenSize1))),
        'h13': tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize3], stddev=np.sqrt(2.0 / hiddenSize1))),

        'h23': tf.Variable(tf.truncated_normal([hiddenSize2, hiddenSize3],stddev=np.sqrt(2.0/hiddenSize2))),


        'out': tf.Variable(tf.truncated_normal([hiddenSize3, ycol],stddev=np.sqrt(2.0/hiddenSize3)))
    }
    biases = {
        'b1': tf.Variable(tf.ones([hiddenSize1])*0.1),
        'b2': tf.Variable(tf.ones([hiddenSize2])*0.1),
        'b3': tf.Variable(tf.ones([hiddenSize3])*0.1),
        'out': tf.Variable(tf.ones([ycol])*0.1)
    }
    ###################|'Function'|scope|###################################
    def add_layer(name, l):
        shape = l.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            c = BatchNorm('bn1', l)
            c = tf.nn.relu(c)
            c = conv('conv1', c, self.growthRate, 1)
            l = tf.concat([c, l], 3)
        return l
    def DenseNet(x, )

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
        
        yout=tf.nn.relu6(tf.matmul(L3,weights['out'])+biases['out'])/6
        return yout
    ###################|'DNN'|scope|######################################
    x=tf.placeholder(tf.float32, [None,xcol])
    y=tf.placeholder(tf.float32, [None,ycol])
    learn_rate = tf.placeholder(tf.float32)
    isTrain=tf.placeholder(tf.bool)
    
    yout=GDNN(x, weights, biases)
    loss = tf.reduce_mean(tf.square(tf.subtract(yout,y)))
    #train = tf.train.RMSPropOptimizer(learn_rate,0.9).minimize(loss)
    train = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    ###################|'predict'|scope|###################################
    _,NNAS=tf.nn.top_k(yout, k=K)
    _,ytestAS=tf.nn.top_k(ytest, k=K)
    NNAS,_=tf.nn.top_k(NNAS,k=K)
    ytestAS,_=tf.nn.top_k(ytestAS,k=K)
    test_prediction = tf.equal(NNAS,ytestAS)
    testAccuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))
    ###################|'train'|scope|####################################################
    init = tf.global_variables_initializer()#tf.initialize_all_variables() #
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)  
        ckpt = tf.train.get_checkpoint_state(mPath)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training from the model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        i=1
        while i <= maxIter:                    
            if i%200==0:
                temp=np.random.randint(1,5)
                DataPath='data10w/data'+str(temp)
                trainH=sio.loadmat(DataPath+'/trainH.mat')
                trainAS=sio.loadmat(DataPath+'/trainAS.mat')
                xtrain=trainH['trainH']
                ytrain=trainAS['trainAS']
            index=np.random.randint(0,numData-1,batch)
            sess.run(train, feed_dict={x: xtrain[index],
                                       y: ytrain[index],
                                       isTrain:True,
                                       learn_rate:0.0001*(0.9**(i/10000))
                                       })
            if i%200==0:
                train_loss = sess.run(loss,feed_dict={x:xtrain[index],
                                                      y:ytrain[index],
                                                      isTrain:True,
                                                      })
                print(i,train_loss)
            ####################################################################################################
            if i%1000==0:
                testAcc=sess.run(testAccuracy,feed_dict={x:xtest,isTrain:False})
                test_loss = sess.run(loss,feed_dict = {x:xtest,y:ytest,isTrain:False})
                print ('###########################')
                print ('testAccuracy:',testAcc)
                print ('testLoss:', test_loss)
                print ('###########################')
            #######################################################################################################
            if i%10000==0:
                print ('--------save model-----------')
                saver.save(sess,mPath+"/model.ckpt")
                print ('saver model')
                print ('-----------------------------')     
            i+=1
        NNAS,ytestAS=sess.run([NNAS,ytestAS],feed_dict={x: xtest,isTrain:False})
        for i in range(numtest):
            H1=np.matrix(H[:,NNAS[i],i])
            H2=np.matrix(H[:,ytestAS[i],i])
            eigv1,_=np.linalg.eig(H1.H*H1)
            eigv2,_=np.linalg.eig(H2.H*H2)
            snr1=snr1+np.max(np.real(eigv1))
            snr2=snr2+np.max(np.real(eigv2))
        print ('---------------------------')
        print ('snr1:',snr1/numtest)
        print ('snr2:',snr2/numtest)
        print ('snr2:',snr1/snr2)
        print ('---------------------------')
if __name__ == '__main__':
    Train()
##tic=time.time()
##toc=time.time()
##print 'time:',toc-tic
##scio.savemat(yout.mat, {'yout':data['yout']})
