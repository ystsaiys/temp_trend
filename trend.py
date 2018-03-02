from util import *
x,y,test = get_data()
train_ids = x.index
test_ids = test.index
x_all = pd.concat([x,test],axis=0) 
x_all = x_all.fillna(-1)
#best_gmm = best_gmm_cluster(x_all)
#joblib.dump(best_gmm,'export/trend_best_gmm.pkl')
best_gmm = joblib.load('trend_best_gmm.pkl')
x_all = best_gmm.predict_proba(x_all)
x = x_all[:x.shape[0]]
test = x_all[:test.shape[0]]


lr = LogisticRegression(C=0.1)
n_estimators,max_depth = 1500,3
xgbm = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth)
lgbm = LGBMClassifier(n_estimators=n_estimators,max_depth=max_depth)
rdf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
classifiers = [rdf,xgbm,lgbm]
grid = StackingClassifier(classifiers=classifiers, 
                           meta_classifier=lr)

params = {'xgbclassifier__n_estimators': [100],
          'xgbclassifier__max_depth': [3],
          'lgbmclassifier__n_estimators':[100],
          'lgbmclassifier__max_depth': [3],
          #'lasso__alpha': [0.01,0.1,0.5],
          'randomforestclassifier__n_estimators':[100],
          'randomforestclassifier__max_depth': [3],
          'meta-logisticregression__C': [0.1]
          }

grid = GridSearchCV(estimator=grid, 
                    param_grid=params, 
                    cv=5,
                    refit=True)

print('fitting')
grid.fit(x,y)

#joblib.dump(grid, 'export/trend_model.pkl') 

predicted = grid.predict_proba(x)
predicted = list(map(lambda x:x[1],predicted))
print('trian roc: ',roc_auc_score(y,predicted))

predicted = pd.Series(grid.predict_proba(test))
result = pd.concat([ids,predicted],axis=1)
result.to_csv('trend_predict_test.csv',index=False)
