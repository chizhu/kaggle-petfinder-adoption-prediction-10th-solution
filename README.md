# kaggle-petfinder-adoption-prediction-10th-solution
kaggle-petfinder-adoption-prediction-10th-solution


First of all, thanks to Petfinder.my and kaggle for hosting this great competition. And congratulations to the winners!Thanks to my teammates for their efforts.
Here is our solution.
## General
As my team name,Stacking is all you need.
We have 4 group features:
### features one:
* 1）clean breed
```python
 def deal_breed(df):
        if df['Breed1']==df['Breed2']:
            df['Breed2']=0
        if df['Breed1']!=307 & df['Breed2']==307:
            temp=df["Breed1"]
            df['Breed1']=df['Breed2']
            df['Breed2']=temp
        return df
    
```
* 2）res features:
```python
 def get_res_feat(df):
        temp=pd.DataFrame(index=range(1))
        temp['RescuerID']=df['RescuerID'].values[0]
        temp['res_type_cnt']=len(df['Type'].unique())
        temp['res_breed_cnt']=len(df['Breed1'].unique())
        temp['res_breed_mode']=df['Breed1'].mode()
        temp['res_fee_mean']=df['Fee'].mean()
        temp['res_Quantity_sum']=df['Quantity'].sum()
        temp['res_MaturitySize_mean']=df['MaturitySize'].mean()
        temp['res_Description_unique']=len(df['Description'].unique())
        return temp
```
* 3)meta features from  public kernel 
* 4)Description features:
* tfidf+svg
* desc+type+breed+color->tfidf+svg/nmf/lda
* desc+type+breed+color->countvec+svg/lda
* desc->wordbatch+svg
* 5)category_col onehot+svg
* 6)densenet121 extract img features+svg
* 7)state features (external data)
* state population density
* state_rank(according to state population density)
* 8)mean target encode with breed(breed=breed1+breed2) boost 0.007(from 0.463 to 0.470)
* 9)linear model oof
* use 1-7 features to build some leaner model ,get linear model oof.

with thease 101 dim features and multi-classify LGB (BEST SINGLE MODEL LB 0.471)
#### note:
get multi-classify result and optimize  it:
sum([0,1,2,3,4]*prob_matrix)
```python
class_list=[0,1,2,3,4]
pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for ix in range(len(pred_test_y[:,0]))]) 
```
### features two
In the early competition, I forked from the public kernel（lb 0.444）
### features three @zhouqingsongct
use featuretools auto extract features 
### features four 
from my teammate @amgis3(lb 0.470)
### models
and we stacking them with these models:
* LGB *6(multi-classify+regression)
* CAT *2(regression)
* NN *3(multi-classify+regression)
* linear model(regression)

### NN @gmhost
we do not use public Embeddings.we just use train+test desc to pretain a new w2v model(think of that are many chinese and Malay.)
maybe we are wrong  our  best nn is near  lb 0.44 

## structures
![img](https://github.com/chizhu/kaggle-petfinder-adoption-prediction-10th-solution/blob/master/img.jpg)

### optR
The open source optR predicts very few zeros, so I manually divide the smallest part of the value to 0 (0.95*LEN_0) (len_0 is the size of 0 in trainset)
```python

def predict(self, X, coef,len_0):
        X_p = np.copy(X)
        temp = sorted(list(X_p))
        threshold=temp[int(0.95*len_0)-1]
        for i, pred in enumerate(X_p):
            if pred < threshold:
                X_p[i] = 0
            elif pred >= threshold and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p
```
