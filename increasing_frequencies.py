import pandas as pd


from cleverminer import cleverminer

df = pd.read_csv ('w:\\development\\cleverminer\\_data\\accidents.txt ', encoding='cp1250', sep='\t')
df=df[['Manoeuvre','Driver_Age_Band','Driver_IMD','Sex','Area','Journey','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type','Vehicle_Age','Hit_Objects_in','Hit_Objects_off','Casualties','Severity','Year','Highway','District','Police']]

df=df[['Vehicle_Age','Vehicle_Type','Driver_Age_Band','Sex','Journey','Area','Manoeuvre','Vehicle_Location','Road_Type','Speed_limit','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)



clm = cleverminer(df=df,target='Year',proc='CFMiner',
  quantifiers= {'S_Up':10, 'Base':3000},
  cond ={'attributes':[
   {'name': 'Vehicle_Age', 'type': 'seq', 'minlen': 1, 
                                          'maxlen': 5},
   {'name': 'Vehicle_Type', 'type': 'subset', 'minlen': 1, 
                                              'maxlen': 1},
   {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 
                                              'maxlen': 3},
   {'name': 'Driver_IMD', 'type': 'seq', 'minlen': 1, 
                                         'maxlen': 3},
   {'name': 'Sex', 'type': 'subset', 'minlen': 1, 
                                     'maxlen': 1},
   {'name': 'Journey', 'type': 'subset', 'minlen': 1, 
                                         'maxlen': 1},
   {'name': 'Area', 'type': 'subset', 'minlen': 1, 
                                      'maxlen': 1},
   {'name': 'Light', 'type': 'subset', 'minlen': 1, 
                                       'maxlen': 1},
   {'name': 'Manoeuvre', 'type': 'subset', 'minlen': 1, 
                                           'maxlen': 1},
   {'name': 'Vehicle_Location', 'type': 'subset', 'minlen': 1, 
                                                  'maxlen': 1},
   {'name': 'Road_Type', 'type': 'subset', 'minlen': 1, 
                                           'maxlen': 1},
   {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 
                                          'maxlen': 1}
   ], 'minlen':1, 'maxlen':4, 'type':'con'}
 )
clm.print_rulelist()
clm.print_summary()
