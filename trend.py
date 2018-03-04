from util import get_data
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
x,y,test = get_data()
train_ids = x.index
test_ids = test.index
x_all = pd.concat([x,test],axis=0) 
#x_all = x_all.fillna(-1)
#best_gmm = best_gmm_cluster(x_all)
#joblib.dump(best_gmm,'export/trend_best_gmm.pkl')
#best_gmm = joblib.load('trend_best_gmm.pkl')
#x_all = best_gmm.predict_proba(x_all)
#x = x_all[:x.shape[0]]
#test = x_all[x.shape[0]:]


lr = LogisticRegression(C=0.1)
xgbm = XGBClassifier(learning_rate=0.1,reg_alpha=0.1, reg_lambda=1)
lgbm = LGBMClassifier(learning_rate=0.1,reg_alpha=0.1, reg_lambda=1)
rdf = RandomForestClassifier()
classifiers = [rdf,xgbm,lgbm]
classifiers = [xgbm,lgbm]
grid = StackingClassifier(classifiers=classifiers,use_probas=True,average_probas=False,meta_classifier=lr)

params = {'xgbclassifier__n_estimators': [100],
          'xgbclassifier__max_depth': [3],
          'lgbmclassifier__n_estimators':[100],
          'lgbmclassifier__max_depth': [3],
          #'randomforestclassifier__n_estimators':[100],
          #'randomforestclassifier__max_depth': [3],
          'meta-logisticregression__C': [0.1]
          }

grid = GridSearchCV(estimator=grid,
                    param_grid=params,
                    cv=5,
                    refit=True,
                    verbose=3,
                    scoring='roc_auc')

print('fitting')
grid.fit(x,y)

joblib.dump(grid, 'trend_model_tmp.pkl') 


predicted = grid.predict_proba(x)
predicted = list(map(lambda x:x[1],predicted))
print('trian roc: ',roc_auc_score(y,predicted))

predicted = pd.Series(grid.predict_proba(test)[:,1])
predicted.index = test_ids
predicted.to_csv('trend_predict_test.csv')
