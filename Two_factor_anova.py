df0 = pd.read_csv("/content/Twitter_volume_AAPL.csv")
df1 =pd.read_csv("/content/Twitter_volume_AMZN.csv")
df2 = pd.read_csv("/content/Twitter_volume_CRM.csv")
df3 = pd.read_csv("/content/Twitter_volume_CVS.csv")

AK_df = pd.read_csv("/content/Twitter_volume_AAPL.csv")
AK_df['AMZN_value'] = df1.value
AK_df['CRM_value'] = df2.value
#AK_df['CVS_value'] = df3.value
column_list = list(AK_df)
AK_df["R_mean"] = AK_df[column_list].mean(axis=1)
AK_df.head()


Apple_m =AK_df.value.mean()
Amzn_m = AK_df.AMZN_value.mean()
Crm_m = AK_df.CRM_value.mean()
print("Mean of Apple : ",Apple_m)
print("Mean of Amzn : ", Amzn_m)
print("Mean of CRM : ", Crm_m)
overall = (Apple_m + Amzn_m+ Crm_m)/3

# SST
lis = ['value', 'AMZN_value', 'CRM_value']
sst = ((overall-AK_df[lis])**2).sum()
sst

#SSC
ssc_list = [Apple_m, Amzn_m, Crm_m]
for i in ssc_list:
  ssc0 = ssc0+(overall-i)**2
ssc0

# SSB
ssb0 = ((AK_df.R_mean - overall)**2).sum()
ssb0

print(sst,"\n", ssc0,"\n \t", ssb)
err = sst - ssc0 -ssb
print("Error = ",err)
