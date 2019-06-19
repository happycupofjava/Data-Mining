
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import svm

'''
 File Name TEMP_FILE_NAME is used to store 
 the pickdataclass output and splitData2TestTrain as Input. 
'''
TEMP_FILE_NAME = 'test.out'


'''
    Converts the string passed to corresponding integers using ASCII values. 
    ASCII(<LETTER>) - 64
    64=ASCIIVALUE('A')-1
'''
def letter_to_digit_convert(mystr):
    m_list = []
    mystr = mystr.upper()
    for i in mystr:
        if i.isalpha():
            m_list.append(ord(i) - 64)
    return m_list


'''
 Splits the data in the file passed as an argument, based on the class ids given by the 
 letter_to_digit_convert function. It stores the output into TEMP_FILE_NAME.
'''


def pickDataClass(filename, class_ids):
    data = np.genfromtxt(filename, delimiter=',')
    #print(data)
    list_ClassifierCol = []
    for i in class_ids:
        a= np.where(data[0] == i)  # returns index locations of the perticular class
        #print(a)
        list_ClassifierCol.extend(np.array(a).tolist())  # appending columns into a string
        #print(listOfClassifierColumn)
    list_ClassifierCol = [item for sublist in list_ClassifierCol for item in sublist]  # forming a array
    np.savetxt(TEMP_FILE_NAME, data[:, list_ClassifierCol], fmt="%i", delimiter=',')
    #fh = open(TEMP_FILE_NAME,"r")
    #print (fh.read())



'''
 splitData2TestTrain takes arguments filename, number_per_class, test_instances
 split the data into testVector, testLabel, trainVector, trainLabel
 Get list of train instances, test instances, strip them and add into respective matrix.
'''


def splitData2TestTrain(filename, number_per_class, test_instances):
    start, end = test_instances.split(":")
    listTest = list(range(int(start), int(end) + 1))
    listTrain = list((set(list(range(0, number_per_class))) - set(listTest)))
    Training = []
    Test = []
    data = np.genfromtxt(filename, delimiter=',')
    #print("x val",data[1].size)
    for i in range(0, data[0].size, number_per_class):
        templistTest = [x + i for x in listTest]
        templistTrain = [x + i for x in listTrain]
        templistTest.sort()
        templistTrain.sort()
        if len(Test) == 0:
            Test = data[:, templistTest]
        else:
            Test = np.concatenate((Test, data[:, templistTest]), axis=1)
        if len(Training) == 0:
            Training = data[:, templistTrain]
        else:
            Training = np.concatenate((Training, data[:, templistTrain]), axis=1)
    return Test[1:, ], Test[0], Training[1:, ], Training[0]


'''
 Stores the np type array into fileName after stacking label over train. 
'''


def store(trainX, trainY, fileName):
    np.savetxt(fileName, np.vstack((trainY, trainX)), fmt="%i", delimiter=',')


'''
 printAccuracy returns the accuracy comparision from Ytest and calculated label
'''
SVM = svm.SVC()


def printAccuracy(sampleLabel, calculatedLabel):
    err_test_padding = sampleLabel - calculatedLabel
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    return (TestingAccuracy_padding)


'''
 Linear regression:
	Xtest_padding : formed by adding ones to bottom of Xtest
	Xtrain_padding: formed by adding ones to bottom of Xtrain
    Ytrain_Indent : Form array with class label index as 1 other are zero.
		e.g LabelVector = [1,5]
		| 1 1 1 0 0 0 |
		| 0 0 0 1 1 1 | 
	return Accuracy. 
'''


def linear(Xtrain, Xtest, Ytrain, Ytest):
    RowToFill = 0
    A_train = np.ones((1, len(Xtrain[0])))
    #print(A_train)
    A_test = np.ones((1, len(Xtest[0])))
    Xtrain_padding = np.row_stack((Xtrain, A_train))
    Xtest_padding = np.row_stack((Xtest, A_test))
    ele, indx, count = np.unique(Ytrain, return_counts=True, return_index=True)
    #print(ele, indx, count) #[1. 2. 3. 4. 5.] [  0  30  60  90 120] [30 30 30 30 30]
    ele = Ytrain[np.sort(indx)]
    #print(np.sort(indx))
    Ytrain_Indent = np.zeros((int(max(ele)), count[0] * len(ele)))
    #print(Ytrain_Indent)
    #print(Ytrain_Indent.shape) # (5, 150)
    for i, j in zip(count, ele):
        Ytrain_Indent[int(j) - 1, RowToFill * i:RowToFill * i + i] = np.ones(i)
        RowToFill += 1
    #print(Ytrain_Indent)
    #print(Xtrain_padding.T)
    B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain_Indent.T)
    Ytest_padding = np.dot(B_padding.T, Xtest_padding)
    #print(Ytest_padding)
    Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1
    #print(Ytest_padding_argmax)
    err_test_padding = Ytest - Ytest_padding_argmax
    #print(err_test_padding)
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    return TestingAccuracy_padding


'''
 kNearestNeighbor(X_train, y_train, X_test, k)
 return y_test
 Find eucledean distance between points, shorlist least k
 returns the dominent label of k.

'''


def kNearestNeighbor(Xtrain, Ytrain, Xtest, k):
    preds = []
    for i in range(len(Xtest[0])):
        t_Test = Xtest[:, i]
        dist = []
        target_class = []
        for iNN in range(len(Xtrain[0])):
            temp_square=np.square(t_Test - Xtrain[:, iNN])
            temp=np.sum(temp_square)
            d = np.sqrt(temp)
            dist.append([d, iNN])
        dist = sorted(dist)
        for iNN in range(k):
            index = dist[iNN][1]
            target_class.append(Ytrain[index])
            #print("@@@",target_class)    
        preds.append(max(set(target_class), key=target_class.count))
        #print(preds)
    preds = list(int(i) for i in preds)
    return preds


'''
 Using the Scikit library sklearn to  to implement the svm method on the data.
'''


def svmClassifier(train, trainLabel, test, testLabel):
    train = train.transpose()
    test = test.transpose()
    SVM.fit(train, trainLabel)
    SVM.predict(test)
    #print(test)
    return test


'''
 centroid method compares the eucledean distance between the 
 nearest centroid. 
'''


def centroid(trainV, trainL, testV):
    temp_arr = []
    #print(len(trainV[0]))
    #print("$$",trainL)
    res = []
    for j in range(0, len(trainV[0]), 8):
        colMean = []
        colMean.append(trainL[j])
        #print(colMean)
        for i in range(len(trainV)):
            x=np.mean(trainV[i, j:j + 7])
            colMean.append(x)
            #print("trainV[i, j:j + 7]:",trainV[i, j:j + 7])
            #print(colMean)
        if not len(temp_arr):
            temp_arr = np.vstack(colMean)
            #print(temp_arr)
        else:
            tempx =np.vstack(colMean)
            temp_arr = np.hstack((temp_arr, tempx))
        #print("$$",temp_arr)
    #print("testV[0]:",len(testV[0]))
    for jN in range(len(testV[0])):
        distances = []
        for m in range(len(temp_arr[0])):
            temp=np.square(testV[:, jN] - temp_arr[1:, m])
            temp=np.sum(temp)
            eucleadian_dist = np.sqrt(temp)
            distances.append([eucleadian_dist, int(temp_arr[0, m])])
            distances = sorted(distances, key=lambda distances: distances[0])
        res.append(distances[0][1])
    #print("Cnetroid result",res)
    return res


'''
 General task function for the Task C and Task D.
'''


def TaskC(uString):
    l_convert = letter_to_digit_convert(uString)
    pickDataClass('HandWrittenLetters.txt', l_convert)
    c_AccList = []
    knnAccList = []
    linearAccList = []
    print('Calculating ' + '.' * 3 )
    for i in range(5, 39, 5):
        #print(str(i))
        testV, testL, trainV, trainL = splitData2TestTrain(TEMP_FILE_NAME, 39, str(i) + ':38')
        c_result = centroid(trainV, trainL, testV)
        c_AccList.append(printAccuracy(testL, c_result))
    X = ['', '(5, 34)', '(10,29)', '(15,24)', '(20,19)', '(25,24)', '(30,9)', '(35,4)']
    fig = plt.figure()
    x1 = fig.add_subplot(111)
    x1.set_xticklabels(X, minor=False)
    x1.set_xlabel('(Train, Test)')
    x1.set_ylabel('Accuracy (%)')
    x1.plot(c_AccList, 'ro', color='black')
    x1.plot(c_AccList, color='grey')
    x1.set_title('Centroid Classification.')
    for i, j in zip(range(7), c_AccList):
        x1.annotate("%.2f" % j, xy=(i + 0.2, j))
    plt.show()


def main():
    choice= input("Please enter which task to be performed(Task A, Task B, Task C or Task D):")
    #raw_input("Please enter which task to be performed(Task A, Task B, Task C):")
    if choice== 'A' or choice == 'a' or choice =='Task A':
        # Task A
        '''TASK A :
                Use the data-handler to select "A,B,C,D,E" classes from the hand-written-letter data.
                From this smaller dataset, Generate a training and test data: for each class.
                using the first 30 images for training and the remaining 9 images for test.
                Do classification on the generated data using the four classifers.'''
        print("Performing Task A\n\n\n")

        pickDataClass('HandWrittenLetters.txt', letter_to_digit_convert('ABCDE')) # classes for the test
        testVector, testLabel, trainVector, trainLabel = splitData2TestTrain(TEMP_FILE_NAME, 39, '30:38') #data split ratio
        svmMatrix = svmClassifier(trainVector, trainLabel, testVector, testLabel)
        c_res = centroid(trainVector, trainLabel, testVector)
        linear_res = linear(trainVector, testVector, trainLabel, testLabel)
        knn_res = kNearestNeighbor(trainVector, trainLabel, testVector, 5)
        svm_res = SVM.score(svmMatrix, testLabel)
        svm_res *= 100
        c=printAccuracy(testLabel, c_res)
        print('\n\nAccuracy of SVM is %0.2f \n' % svm_res)
        print('Accuracy of Centroid is %0.2f\n' % c)
        print('Accuracy of Linear is %0.2f\n' % linear_res)
        v=printAccuracy(testLabel, knn_res)
        print('Accuracy of 5-NN is %0.2f\n' % v)

    elif choice== 'B' or choice == 'b' or choice =='Task B':

        # Task B
        '''TASK B 
                On ATNT data, run 5-fold cross-validation (CV) using  each of the
                four classifiers: KNN, centroid, Linear Regression and SVM.
                If you don't know how to partition the data for CV, you can use the data-handler to do that.
                Report the classification accuracy on each classifier.
                Remember, each of the 5-fold CV gives one accuracy. You need to present all 5 accuracy numbers
                for each classifier. Also, the average of these 5 accuracy numbers.'''
        print("Performing Task B\n\n\n")
        svmAccList = []
        centroidAccList = []
        knnAccList = []
        linearAccList = []
        print('Calculating' + '.' * 3)
        for i in range(0, 10, 2):
            testVector, testLabel, trainVector, trainLabel = splitData2TestTrain('ATNTFaceImages400.txt', 10,
                                                                                 str(i) + ':' + str(i + 1))
            #change the  split ratio: change for in range(0,10, taining numebr) and i+test-1
            """
            print("",testVector)
            print(testLabel)
            print(trainVector)
            print(trainLabel)
            print("data",str(i), str(i+1))"""
            svm_Matrix = svmClassifier(trainVector, trainLabel, testVector, testLabel)
            centroid_res = centroid(trainVector, trainLabel, testVector)
            linearAccList.append(linear(trainVector, testVector, trainLabel, testLabel))
            knn_res = kNearestNeighbor(trainVector, trainLabel, testVector, 5)
            svm_res = SVM.score(svm_Matrix, testLabel)
            svm_res *= 100
            svmAccList.append(svm_res)
            centroidAccList.append(printAccuracy(testLabel, centroid_res))
            knnAccList.append(printAccuracy(testLabel, knn_res))
        knn_r=sum(knnAccList) / len(knnAccList)
        print('\nAverage accuracy of 5-NN after 5-Fold is %0.2f' % knn_r)
        print(knnAccList)
        centroid_r=sum(centroidAccList) / len(centroidAccList)
        print('\nAverage accuracy of Centroid after 5-Fold is %0.2f' % centroid_r)
        print(centroidAccList)
        svm_r=sum(svmAccList) / len(svmAccList)
        print('\nAverage accuracy of SVM after 5-Fold is %0.2f' % svm_r)
        print(svmAccList)
        linear_r=sum(linearAccList) / len(linearAccList)
        print('\nAverage accuracy of Linear after 5-Fold is %0.2f' % linear_r)
        print(linearAccList)

    elif choice== 'C' or choice == 'c' or choice =='Task C':
        # Task C
        ''' TASK C : 
                On handwritten letter data, fix on 10 classes. Use the data handler to generate training and test data files.
                Do this for seven different splits:  (train=5 test=34), (train=10 test=29),  (train=15 test=24) ,
                (train=20 test=19), (train=25 test=24) , (train=30 test=9) ,  (train=35 test=4). 
                On these seven different cases, run the centroid classifier to compute average test image classification
                accuracy. Plot these 7 average accuracy on one curve in a figure. What trend can you observe?
                When do this task, the training data and test data do not need be written into files.'''

        print("Performing Task C\n\n\n")
        TaskC('ABCDEFXYZG')

    elif choice== 'D' or choice == 'd' or choice =='Task D':
        ''' TASK D:
                Repeat task (D) for another different 10 classes.  You get another 7 average accuracy.
                Plot them on one curve in the same figure as in task (D). Do you see some trend?'''
        print("Performing Task D\n\n\n")
        TaskC('MNOPQRSTUV')

    else:
        print( "pick one of the tasks in A, B c or D only!!!\n\n\n")
        main()

main()
