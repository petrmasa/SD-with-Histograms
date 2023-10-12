import pandas as pd


from cleverminer import cleverminer

df = pd.read_csv ('w:\\development\\cleverminer\\_data\\accidents.txt ', encoding='cp1250', sep='\t')
df=df[['Manoeuvre','Driver_Age_Band','Driver_IMD','Sex','Area','Journey','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type','Vehicle_Age','Hit_Objects_in','Hit_Objects_off','Casualties','Severity','Year','Highway','District','Police']]

df=df[['Vehicle_Age','Vehicle_Type','Driver_Age_Band','Sex','Journey','Area','Manoeuvre','Vehicle_Location','Road_Type','Speed_limit','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)


