import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

def get_vif(X,sort_result=True):
	vif_df = pd.DataFrame()
	vif_df['Features'] = X.columns
	vif_df['VIF'] = [VIF(X.values,i) for i in range(X.shape[1])]
	vif_df['VIF'] = round(vif_df['VIF'],2)
	if(sort_result):
		vif_df = vif_df.sort_values(by='VIF',ascending=False)
	return vif_df