import pandas as pd

#retrieving an excel file from the end assessment folder
mushroom_excel_data = pd.read_excel('end assessment/mushroom_species.xlsx', sheet_name='mushroom')

#convert the file into a csv file)
mushroom_excel_data.to_csv('end assessment/mushroom_species.csv', index=None, header=True)