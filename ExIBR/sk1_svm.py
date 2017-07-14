from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm,grid_search
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn import metrics
from numpy import genfromtxt
import scipy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
from scipy import stats

def read_data(filename):
    
    my_data = genfromtxt(filename, delimiter=',')    
    return my_data;
    
def main():
    print str(datetime.now())
    print 'svm'
    precision=[0]*10;
    recall=[0]*10;
    fmeasure=[0]*10;

    precision1=[0]*10;
    recall1=[0]*10;
    fmeasure1=[0]*10;

    precision2=[0]*10;
    recall2=[0]*10;
    fmeasure2=[0]*10;

    
    
    for k in range (78,82):
        #(36,37):
        #(42,43):
        cc=0;
        gg=0;
        if k==0:
            type='word';
            cc=64;
            gg=0.0625;
        if k==1:
            type='user_lda';
            cc=2;
            gg=1;
        if k==2:
            type='sc_lda';
            cc=32;
            gg=0.25;
        if k==3:
            type='liwc';
            cc=2;
            gg=1;
        if k==4:
            type='posting';
            cc=16;
            gg=0.0625;
        if k==5:
            type='writing';
            cc=32;
            gg=1;
        if k==6:
            type='demo';
            cc=8;
            gg=1;
        if k==12:
            type='LDA_F';
            cc=0.5;
            gg=4;
        if k==13:
            type='LDA_I';
            cc=0.125;
            gg=16;
        if k==14:
            type='LDA_TF';
            cc=2;
            gg=1;
        if k==15:
            type='LDA_TI';
            cc=4;
            gg=0.25;
        if k==16:
            type='LDA_FI';
            cc=0.5;
            gg=4;
        if k==17:
            type='LDA_TIF';
            cc=2;
            gg=0.5;
        if k==18:
            type='LIWC_T';
            cc=2;
            gg=2;
        if k==19:
            type='LIWC_F';
            cc=1;
            gg=16;
        if k==20:
            type='LIWC_I';
            cc=2;
            gg=2;
        if k==21:
            type='LIWC_TF';
            cc=1;
            gg=16;
        if k==22:
            type='LIWC_TI';
            cc=2;
            gg=1;
        if k==23:
            type='LIWC_IF';
            cc=1;
            gg=16;
        if k==24:
            type='LIWC_TIF';
            cc=1;
            gg=16;
        if k==25:
            type='T';
            cc=16;
            gg=0.0625;
        if k==26:
            type='F';
            cc=2;
            gg=1;
        if k==27:
            type='I';
            cc=0.25;
            gg=0.0625;
        if k==28:
            type='TF';
            cc=16;
            gg=0.125;
        if k==29:
            type='TI';
            cc=32;
            gg=0.0625;
        if k==30:
            type='IF';
            cc=16;
            gg=0.125;
        if k==31:
            type='TIF';
            cc=32;
            gg=0.0625;
        if k==32:
            type='demo';
            cc=32;
            gg=1;
        if k==33:
            type='pa';
            cc=4;
            gg=0.125;
        if k==34:
            type='EX_USER_LDA';
            cc=32;
            gg=0.0625;
        if k==35:
            type='EX_DC';
            cc=16;
            gg=0.0625;
        if k==36:
            type='EX_LIWC';
            cc=16;
            gg=0.125;
        if k==37:
            type='EX_BEHA';
            cc=8;
            gg=0.25;
        if k==38:
            type='EX_SC_LDA';
            cc=16;
            gg=0.0625;
        if k==39:
            type='EX_WORD';
            cc=16;
            gg=0.0625;
        if k==40:
            type='EX_POST';
            cc=8;
            gg=0.25;
        if k==41:
            type='EX_EGO';
            cc=8;
            gg=0.125;
        if k==42:
            type='EX_PLINGUSTIC';
            cc=32;
            gg=0.0625;
        if k==43:
            type='beha';
            cc=32;
            gg=0.0625;
        if k==44:
            type='linguistic';
            cc=2;
            gg=0.5;
        if k==45:
            type='pa';
            cc=16;
            gg=0.0625;
        if k==46:
            type='EX_PA';
            cc=8;
            gg=1;
        if k==47:
            type='EX_LINGUISTIC';
            cc=16;
            gg=0.0625;
        if k==48:
            type='aggre_miss_total4_replace5total1';
            cc=4;
            gg=0.125;
        if k==49:
            type='aggre_miss_total4_replace10total1';
            cc=4;
            gg=0.25;
        if k==50:
            type='aggre_miss_total4_replace15total1';
            cc=8;
            gg=0.0625;
        if k==51:
            type='aggre_miss_total4_replace20total1';
            cc=4;
            gg=0.0625;
        if k==52:
            type='aggre_miss_total4_replace25total1';
            cc=8;
            gg=0.0625;
        if k==53:
            type='aggre_miss_total4_replace30total1';
            cc=4;
            gg=0.0625;
        if k==54:
            type='aggre_miss_total4_replace35total1';
            cc=16;
            gg=0.0625;
        if k==55:
            type='aggre_miss_total4_replace40total1';
            cc=8;
            gg=0.125;
        if k==56:
            type='aggre_miss_total4_replace45total1';
            cc=4;
            gg=0.125;
        if k==57:
            type='aggre_miss_total4_replace50total1';
            cc=32;
            gg=0.0625;
        if k==58:
            type='realusers';
            cc=16
            gg=0.0625;
        if k==59:
            type='D';
            cc=32
            gg=0.0625;
        if k==60:
            type='P';
            cc=16
            gg=0.0625;
        if k==61:
            type='V';
            cc=16
            gg=0.25;
        if k==62:
            type='DV';
            cc=2;
            gg=0.5;
        if k==63:
            type='DP';
            cc=16
            gg=0.0625;
        if k==64:
            type='PV';
            cc=16
            gg=0.125;
        if k==65:
            type='DVP1';
            cc=8
            gg=0.125;
        if k==66:
            type='PLINGUISTIC';
            cc=1
            gg=0.5;
        if k==67:
            type='BEHA';
            cc=16
            gg=0.0625;
        if k==68:
            type='posting';
            cc=32
            gg=0.25;
        if k==69:
            type='EGO';
            cc=32
            gg=1;
        if k==70:
            type='NX_notfix_k30_lam1_beta100_5';
            cc=32
            gg=0.0625;
        if k==71:
            type='follow_lda';
            cc=32
            gg=0.5;
        if k==72:
            type='retweet_lda';
            cc=8
            gg=1;
        if k==73:
            type='post_wr';
            cc=32
            gg=1;
        if k==74:
            type='post_ca';
         #   cc=4
          #  gg=0.0625;
            cc=20
            gg=0.5
        if k==75:
            type='post_in';
          #  cc=0.25
           # gg=0.5;
            cc=0.5;
            gg=0.5
        if k==76:
            type='EX_follow_lda';
            cc=16
            gg=0.0625;
        if k==77:
            type='EX_retweet_lda';
            cc=16
            gg=0.0625;
        if k==78:
            type='EX_post_wr';
            cc=4
            gg=0.125;
#            cc=16
 #           gg=0.125;
        if k==79:
            type='EX_post_ca';
            cc=32
            gg=0.0625;
   #         cc=16
    #        gg=0.0625;
        if k==80:
            type='EX_post_in';
            cc=8
            gg=0.125;
      #      cc=8
       #     gg=0.125;
        if k==81:
            type='EX_POST';
            cc=8;
            gg=0.25;
            
        print type
        for i in range (0,10):
            print i;
            traindata=read_data('D:/py1/Sub/'+type+'/'+str(i)+type+'train_data_scale.csv');
       #     traindata=read_data('D:/py1/Sub/'+type+'/'+str(i)+'train_data_scale.csv');
            [row, col]= np.shape(traindata);
       
            X_train=traindata[0:row,:col-2]; #[1:row,:col-2];
            y_train=traindata[0:row,col-1];
     

            testdata=read_data('D:/py1/Sub/'+type+'/'+str(i)+type+'test_data_scale.csv');
        #    testdata=read_data('D:/py1/Sub/'+type+'/'+str(i)+'test_data_scale.csv');
            [row, col]= np.shape(testdata);
         
            X_test=testdata[0:row,:col-2];
            y_test=testdata[0:row,col-1];


            svr= svm.SVC()
            clf=svm.SVC(C=cc, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=gg, kernel='rbf', probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False).fit(X_train, y_train);
            y_pred= clf.predict(X_test);
            
            prob= clf.predict_proba(X_test);
            filename='E:/aboutme_linkedin/LDA/spilit/SVM/prediction_svm'+str(i)+'.txt';
            np.savetxt(filename, prob, delimiter=',')
            
            result=precision_recall_fscore_support(y_test, y_pred)
            precision[i]=round(result[0][0]*100, 2);
            
            recall[i]=round(result[1][0]*100, 2);
            fmeasure[i]=round(result[2][0]*100, 2);

            precision1[i]=round(result[0][1]*100, 2);
            recall1[i]=round(result[1][1]*100, 2);
            fmeasure1[i]=round(result[2][1]*100, 2);

            result=precision_recall_fscore_support(y_test, y_pred, average='weighted')
            precision2[i]=round(result[0]*100, 2);
            recall2[i]=round(result[1]*100, 2);
            fmeasure2[i]=round(result[2]*100, 2);

            print fmeasure2[i];
            
            result1=accuracy_score(y_test, y_pred)
  #          print 'y_test'
   #         print y_test
    #        print 'y_pred'
     #       print y_pred
      #      print 'result1'
       #     print result1
        print 'precision2:'
        print precision2
        print 'recall2:'
        print recall2
        print 'fmeasure2:'
        print fmeasure2
        me= np.mean(precision2);
        ttest(precision2, me, 'precision2');
        me= np.mean(recall2);
        ttest(recall2, me,'recall2');
        me=np.mean(fmeasure2)
        ttest(fmeasure2, me,'fmeasure2');
        print str(datetime.now())    

def ttest(ar, mea, type):
    one_sample = stats.ttest_1samp(ar, mea)
    print type+ "The t-statistic is %.3f and the p-value is %.3f." % one_sample
    print mea

    
if __name__ == '__main__':
    main()

