# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:58:13 2023

@author: kellyj2
    
  
"""

### Import Packages
import pandas as pd
from os import listdir, getlogin

### Load in ESS June
ess_june = pd.read_csv("C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Dissemination/X_Govt_Dashboard/Automation/ESS pipeline June/Outputs/for IDS/ESS_June23_data.csv")

### Create file paths
input_path = "//TS003/kellyj2$/My Documents"
mapper_name = "geography_mapper_v4_2.csv"

### Define Geomapper Functions 
def load_geography_mapper(
    input_path: str,
    mapper_name: str,
    encoding_type: str = "ISO-8859-1",
) -> pd.DataFrame:
    """Loads the mapping of area code to geography.
    Parameters
    ----------
    input_path
        Path to where the mapper is stored.
    mapper_name
        Name of the geography mapper file.
    encoding_type
        type of encoding to ensure that utf-8 is not an error
        default = ISO-8859-1

    Returns
    -------
    DataFrame containing mapping from area code to geography.
    """
    mapper = pd.read_csv(f"{input_path}/{mapper_name}", encoding = encoding_type)

    return mapper


def apply_geography_mapper(
    df: pd.DataFrame,
    mapper: pd.DataFrame,
) -> pd.DataFrame:
    """Applies geography mapper. Orders by geography.
    Needs to get geographies into the right place. Depends on formatting of input mapper.
    Parameters
    ----------
    df
        Contains the loaded data.
    mapper
        DataFrame containing mapping from area code to geography.
    Returns
    -------
    DataFrame with area codes changed to geography.
    """
    df, mapper = get_area_codes_from_mapper(df, mapper)

    # Finds column where there is a value and assigns to 'Geography' column
    mapper_edited = mapper.drop(columns=['Area codes']).notna()
    mapper_edited['Geography'] = mapper_edited.apply(lambda r: r.idxmax(skipna=True), axis=1)
    mapper_edited = pd.concat([mapper['Area codes'], mapper_edited['Geography']], axis=1)

    # Get area codes that are in both mapper and data
    mapper_edited = mapper_edited[mapper_edited['Area codes'].isin(df['AREACD'])]
    df = mapper_edited.merge(df, right_on="AREACD", left_on="Area codes", how="right", sort=False)
    df = df.drop(columns='Area codes')

    df = mapper.merge(df, right_on="AREACD", left_on="Area codes", how="right", sort=False)
    df = df.drop(columns='Area codes')

    # Overwrite df AREANM those from geography mapper after applying it
    geo_cols = df["Geography"].dropna().unique()
    geog_order = list(df.columns)
    geo_cols = sorted(
        geo_cols,
        key=lambda e: (geog_order.index(e), e) if e in geog_order else (len(geog_order), e)
    )
    flat_series_df = df[geo_cols].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    flat_series_df.name = "AREANM"
    df['AREANM'] = flat_series_df

    # Get area names from mapper
    geog_order = list(mapper.columns)
    geo_cols = sorted(
        geo_cols,
        key=lambda e: (geog_order.index(e), e) if e in geog_order else (len(geog_order), e)
    )
    mapper['AREANM'] = mapper[geo_cols].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    flat_series_mapper = mapper[['Area codes', 'AREANM']]
    flat_series_mapper = flat_series_mapper.apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    flat_series_mapper.name = "SORTER"

    # Sort by AREACD/AREANM
    df['SORTER'] = df[['AREACD', 'AREANM']].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    df['SORTER'] = pd.Categorical(df['SORTER'], categories=flat_series_mapper.unique().tolist(), ordered=True)
    df = df.sort_values(by="SORTER")

    df.loc[df['AREACD'].str[:2] == 'na', 'AREACD'] = 'na'

    # Overwrite AREACD and AREANM with replacement columns from mapper.
    # Performance would greatly improve from being vectorized
    area_name_col_index = df.columns.get_loc('AREANM')
    # Gets column indexes that need to be replaced with new area names
    df['col_indexes'] = df.apply((lambda x: [idx for idx, elem in enumerate(x) if elem == x[area_name_col_index]]), axis=1)

    # Iterate over rows and columns that need to be replaced and replace with value from Area Name Replace
    area_name_replace_col_index = df.columns.get_loc('Area Name Replace')
    for row in range(len(df.index)):
        cols_to_replace = df.iloc[row, -1]
        for col_index in cols_to_replace:
            df.iloc[row, col_index] = df.iloc[row, area_name_replace_col_index]

    df = df.drop(columns=['AREACD'])
    df = df.rename(columns={
        "Area Code Replace": "AREACD",
    })

    return df


def get_area_codes_from_mapper(
    df: pd.DataFrame,
    geography_mapper: pd.DataFrame,
) -> pd.DataFrame:
    """Gets area codes when the AREACD column is blank

    Parameters
    ----------
    df
        Contains the loaded data.
    geography_mapper
        Geography mapper.

    Returns
    -------
    DataFrame with area codes added.
    Mapper with na values made unique.
    """
    # Assign na values unique numbers
    mask = geography_mapper['Area codes'] == 'na'
    geography_mapper['numbers'] = range(geography_mapper.shape[0])
    geography_mapper.loc[mask, 'Area codes'] = (
            geography_mapper.loc[mask, 'Area codes'] +
            geography_mapper.loc[mask, 'numbers'].astype(str)
    )
    geography_mapper = geography_mapper.drop(columns="numbers")

    # Add area codes from geography mapper
    new_cols = [col for col in geography_mapper.columns if (
            (col != "Area codes") &
            (col != "Notes") &
            (col != "Area Code Replace") &
            (col != "Area Name Replace"))]
    geocode_mapper = geography_mapper.copy()
    flat_series = geocode_mapper[new_cols].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    flat_series.name = "AREANM"
    geoportal_codes = pd.concat([geocode_mapper['Area codes'], flat_series], axis=1)
    geoportal_codes = geoportal_codes.rename(columns={"Area codes": "AREACD"})
    df_temp = df.loc[df['AREACD'].isna()]
    if df_temp.shape[0] > 0:
        df_temp = df_temp.merge(geoportal_codes, left_on=["AREANM"], right_on=["AREANM"], how="left")
        df_temp = df_temp.drop(['AREACD_x'], axis=1)
        df_temp = df_temp.rename(columns={'AREACD_y': 'AREACD'})
        df = pd.concat([df, df_temp])

    return df, geography_mapper


input_path = "\\TS003\kellyj2$\My Documents\geography_mapper_v4_2.csv"



#############################################################################################################
"""
    Active Enterprises 
    
"""

### File name
file_name = "Active enterprises"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

# Capitalise 
df["AREANM"] = df["AREANM"].str.title()

# ### Decapitalise variables and remove *
# df["AREANM"] = [c.replace("And", "and") for c in df["AREANM"]]
# df["AREANM"] = [c.replace("The", "the") for c in df["AREANM"]]
# df["AREANM"] = [c.replace("On", "on") for c in df["AREANM"]]
# df["AREANM"] = [c.replace("Of", "of") for c in df["AREANM"]]
# df["AREANM"] = [c.replace("Upon", "upon") for c in df["AREANM"]]
# df["AREANM"] = [c.replace("*", "") for c in df["AREANM"]]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:12], value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Select Columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Number of active enterprises in the UK by year" 

# Add "Indicator" column
df_long["Indicator"] = "Active Enterprises"

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Boosting productivity, pay, jobs and living standards"

# Add "Measure" column
df_long["Measure"] = "Count"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")

#############################################################################################################
"""
    Business Births 
    
"""

### File name
file_name = "Business Births"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:12], value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# 2688
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Number of business births in the UK by year" 

# Add "Indicator" column
df_long["Indicator"] = "Business Births"

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Boosting productivity, pay, jobs and living standards"

# Add "Measure" column
df_long["Measure"] = "Count"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
    Age standardised suicide rate
    
    
"""

### File name
file_name = "Age standardised suicide rate"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

##############################################
"""
    Suicide: Raw number of deaths 
    
"""

### Create list of cols for Number of Deaths
col_list = list(df.columns)
item_index = list(range(2, 40, 2)) # Get columns in DF
num_of_deaths = [col_list[i] for i in item_index]

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = num_of_deaths, value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

### Remove suffix from Period
df_long["Period"] = [c.replace(" Number of deaths", "") for c in df_long["Period"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Raw number of deaths from Suicide" 

# Add "Indicator" column
df_long["Indicator"] = "Suicide deaths" 

# Add "Mission" column
df_long["Mission"] = "Mission 8"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Number"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}_raw.csv")


##############################################
"""
    Suicide: Rate per 100,000
    
"""

### Create list of cols for Number of Deaths
col_list = list(df.columns)
item_index = list(range(3, 41, 2)) # Get columns in DF
rate_per_100k = [col_list[i] for i in item_index]

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = rate_per_100k, value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Remove suffix from Period
df_long["Period"] = [c.replace(" Rate per 100,000", "") for c in df_long["Period"]]

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Death rate per 100,000 from Suicide" 

# Add "Indicator" column
df_long["Indicator"] = "Suicide rate per 100,000"

# Add "Mission" column
df_long["Mission"] = "Mission 8"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Rate per 100,000 population"

# Add "Unit" column
df_long["Unit"] = "Rate per 100,000 population"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}_rate_100k.csv")



#############################################################################################################
"""
    Hospital admisions due to knife crime 
    
"""

### File name
file_name = "Hospital admissions due to knife crime"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")
# lad_changes = pd.read_excel("//TS003/kellyj2$/Desktop/IDS/Subnational Statistics/Data/Geo Look-ups/LAD changes_2012_2023.xlsx", sheet_name = "England LAD changes 2012-21")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:12], value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Number of hospital admissions due to knife crime" 

# Add "Indicator" column
df_long["Indicator"] = "Hospital admissions due to knife crime"

# Add "Mission" column
df_long["Mission"] = "Mission 11"

# Add "Category" column
df_long["Category"] = "Restoring a sense of community, local pride and belonging"

# Add "Measure" column
df_long["Measure"] = "Count"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")

#############################################################################################################
"""
    Hospital admissions due to violence 
    
"""

### File name
file_name = "Hospital admissions due to violence"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Drop "Parent Name"
df = df.drop("Parent Name", axis = 1)

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

## Recode "Time period" to "Time"
df = df.rename(columns = {"Year" : "Period"})

### Map Time Variables (Old > New)
time_mapping = {'2009/10 - 11/12' : '04/2009-03/2012', 
                '2010/11 - 12/13' : '04/2010-03/2013', 
                '2011/12 - 13/14' : '04/2011-03/2014',
                '2012/13 - 14/15' : '04/2012-03/2015', 
                '2013/14 - 15/16' : '04/2013-03/2016', 
                '2014/15 - 16/17' : '04/2014-03/2017',
                '2015/16 - 17/18' : '04/2015-03/2018', 
                '2016/17 - 18/19' : '04/2016-03/2019', 
                '2017/18 - 19/20' : '04/2017-03/2020',
                '2018/19 - 20/21' : '04/2018-03/2021'}

### Assign new values
df = df.assign(Period = df.Period.map(time_mapping))

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df["Variable Name"] = "Hospital admissions due to violence rate per 100,000 population" 

# Add "Indicator" column
df["Indicator"] = "Hispital admissions due to violence"

# Add "Mission" column
df["Mission"] = "Mission 11"

# Add "Category" column
df["Category"] = "Restoring a sense of community, local pride and belonging"

# Add "Measure" column
df["Measure"] = "Rate per 100,000 population"

# Add "Unit" column
df["Unit"] = "Rate per 100,000 population"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
         'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")



#############################################################################################################
"""
    Hospital Admissions due to self harm
    
"""

### File name
file_name = "Hospital admissions self harm"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Drop Area Type column
df = df.drop("Area Type", axis = 1)

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

### Recode "Time period" to "Time"
df = df.rename(columns = {"Time period" : "Period"})

### Recode Time Variables (Old > New)
time_mapping = {'2010/11' : '04/2010-03/2011', 
                '2011/12' : '04/2011-03/2012', 
                '2012/13' : '04/2012-03/2013', 
                '2013/14' : '04/2013-03/2014', 
                '2014/15' : '04/2014-03/2015', 
                '2015/16' : '04/2015-03/2016',
                '2016/17' : '04/2016-03/2017', 
                '2017/18' : '04/2017-03/2018', 
                '2018/19' : '04/2018-03/2019', 
                '2019/20' : '04/2019-03/2020', 
                '2020/21' : '04/2020-03/2021', 
                '2021/22' : '04/2021-03/2022'}

### Assign new values
df = df.assign(Period = df.Period.map(time_mapping))

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df["Variable Name"] = "Hospital admissions due to self harm rate per 100,000 population" 

# Add "Indicator" column
df["Indicator"] = "Hospital admissions due to self harm"

# Add "Mission" column
df["Mission"] = "Mission 7"

# Add "Category" column
df["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df["Measure"] = "Rate per 100,000 population"

# Add "Unit" column
df["Unit"] = "Rate per 100,000 population"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
         'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   Median house prices to earnings
    
"""

### File name
file_name = "Median house prices to earnings"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:23], value_name = "Value")\
    .rename(columns = {"variable": "Period"})
    
### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Median house prices to median annual earnings ratio" 

# Add "Indicator" column
df_long["Indicator"] = "Median house prices to median annual earnings"

# Add "Mission" column
df_long["Mission"] = "Mission 10"

# Add "Category" column
df_long["Category"] = "Restoring a sense of community, local pride and belonging"

# Add "Measure" column
df_long["Measure"] = "Ratio"

# Add "Unit" column
df_long["Unit"] = "Ratio"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   Households EPC band C
    
"""

### File name
file_name = "Households EPC band C"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

df = df.drop("REGNM", axis = 1)\
       .drop("REGCD", axis = 1)

### Recode "Time period" to "Time"
df = df.rename(columns = {"All dwellings" : "Value"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography", "Value"]]

# Add "Variable Name" column
df["Variable Name"] = "Percentage of households with EPC band C or above" 

# Add "Indicator" column
df["Indicator"] = "Households with EPC band C or above"

# Add "Mission" column
df["Mission"] = "Mission 10"

# Add "Category" column
df["Category"] = "Restoring a sense of community, local pride and belonging"

# Add "Measure" column
df["Measure"] = "Percentage"

# Add "Unit" column
df["Unit"] = "%"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
         'Mission', 'Category', 'Value', 'Measure', 'Unit']]

### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   Indicies of Deprivation
    
"""

### File name
file_name = "Indices of deprivation"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Recode "Time period" to "Time"
df = df.rename(columns = {"IMD - Rank of average score " : "Rank of Average Score"})

#calculate decile of each value in data frame
df['Decile'] = pd.qcut(df['Rank of Average Score'], 10, labels=False) +1

df = df[["AREACD", "AREANM", "Rank of Average Score", "Decile"]]

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography", "Rank of Average Score", "Decile"]]

# Add "Variable Name" column
df["Variable Name"] = "Indicies of Deprivation" 

# Add "Indicator" column
df["Indicator"] = "Indicies of Deprivation"

# Add "Mission" column
df["Mission"] = "Misc"

# Add "Category" column
df["Category"] = "Misc"

# Add "Measure" column
df["Measure"] = "Rank"

# Add "Unit" column
df["Unit"] = "Decile"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
         'Mission', 'Category', 'Rank of Average Score',  "Decile", 'Measure', 'Unit']]

### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv", index = False)


#############################################################################################################
"""
   Method of travel
    
"""

### File name
file_name = "Method of travel"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

### Recode "Time period" to "Time"
df = df.rename(columns = {"Method used to travel to workplace (12 categories)" : "Travel Category",
                          "Observation" : "Value"})

### Select Variables
df = df[["AREACD", 
         "AREANM", 
         "Travel Category", 
         "Value"]]

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography", "Travel Category", "Value"]]

# Add "Variable Name" column
df["Variable Name"] = "Method of travel to work (Census 2021)" 

# Add "Indicator" column
df["Indicator"] = "Method of travel to work"

# Add "Mission" column
df["Mission"] = "Mission 3"

# Add "Category" column
df["Category"] = "Boosting productivity, pay, jobs and living standards"

# Add "Measure" column
df["Measure"] = "Count"

# Add "Unit" column
df["Unit"] = "Number"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
         'Mission', 'Category', "Travel Category", 'Value', 'Measure', 'Unit']]

### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   ONS Health index
    
"""

### File name
file_name = "ONS Health index basic"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]
df["AREANM"] = [c.replace("ENGLAND", "England") for c in df["AREANM"]]

### Drop "Area Type" Column
df = df.drop("Area Type", axis = 1)

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[3:10], value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "ONS Health Index" 

# Add "Indicator" column
df_long["Indicator"] = "ONS Health Index"

# Add "Mission" column
df_long["Mission"] = "Mission 7"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Value"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   Standard occupation Nomis data
    
"""

### File name
file_name = "Standard occupation Nomis data"
df = pd.read_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.csv", encoding = "latin1")

### Remove \n from column names
df.columns = [c.replace("\n", "") for c in df.columns]

### Recode "Time period" to "Time"
df = df.rename(columns = {"Row Labels" : "AREANM"})
df = df.rename(columns = {"Category" : "Occupation Category"})

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM", "Period"], value_vars = col_list[3:21], value_name = "Value")\
    .rename(columns = {"variable": "Occupation Category"})
    
### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography",  "Period", "Occupation Category", "Value"]]

df_long = df_long.sort_values(by = ["AREANM", "Occupation Category", "Period"])

# Add "Variable Name" column
df_long["Variable Name"] = "Number of people in occupational category" 

# Add "Indicator" column
df_long["Indicator"] = "Occupation by category"

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Boosting productivity, pay, jobs and living standards"

# Add "Measure" column
df_long["Measure"] = "Count"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', "Occupation Category", 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv", index = False)


#############################################################################################################
"""
   Census population: age by sex
    
"""

### File name
file_name = "Census_2021_Age_Sex_population"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Remove \n from column names
df["AREANM"] = [c.replace("[note 3]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 4]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 5]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 6]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 7]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 8]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 9]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 10]", "") for c in df["AREANM"]]
df["AREANM"] = [c.replace("[note 11]", "") for c in df["AREANM"]]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM", "Gender"], value_vars = col_list[4:23], value_name = "Value")\
    .rename(columns = {"variable": "Age"})
    
### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography",  "Gender", "Age", "Value"]]

# Add "Variable Name" column
df_long["Variable Name"] = "Census 2021 population by age and sex" 

# Add "Indicator" column
df_long["Indicator"] = "Census 2021 population by age and sex"

# Add "Mission" column
df_long["Mission"] = "Misc"

# Add "Category" column
df_long["Category"] = "Misc"

# Add "Measure" column
df_long["Measure"] = "Count"

# Add "Unit" column
df_long["Unit"] = "Number"


# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Gender', "Age", 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


#############################################################################################################
"""
   Census: Median Age
    
"""

### File name
file_name = "Census_2021_median_age"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Create list of names
col_list = list(df.columns)

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography",  "Median Age"]]\
    .rename(columns = {"Median Age": "Value"})

# Add "Variable Name" column
df["Variable Name"] = "Census 2021 Median Age" 

# Add "Indicator" column
df["Indicator"] = "Census 2021 Median Age"

# Add "Mission" column
df["Mission"] = "Misc"

# Add "Category" column
df["Category"] = "Misc"

# Add "Measure" column
df["Measure"] = "Median"

# Add "Unit" column
df["Unit"] = "Number"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Value', 'Measure', 'Unit']]
### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv", index = False)

#############################################################################################################
"""
   Census: Median Age
    
"""

### File name
file_name = "Census_2021_Population density"
df = pd.read_excel(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.xlsx", sheet_name= "Sheet1")

### Create list of names
col_list = list(df.columns)

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df = apply_geography_mapper(df, geomapper)

# Drop Geomapper columns
df = df[["AREACD", "AREANM", "Geography",  "Population density"]]\
    .rename(columns = {"Population density": "Value"})

# Add "Variable Name" column
df["Variable Name"] = "Census 2021 Population Density" 

# Add "Indicator" column
df["Indicator"] = "Census 2021 Population Density"

# Add "Mission" column
df["Mission"] = "Misc"

# Add "Category" column
df["Category"] = "Misc"

# Add "Measure" column
df["Measure"] = "per sq. km"

# Add "Unit" column
df["Unit"] = "Number"

# Restructure Columns
df = df[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Value', 'Measure', 'Unit']]
### Save out data
df.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv", index = False)





##############################################################################################################
"""
   Economic activity
    Drop time from Variable name
    
    Add measure/units
    
    Indicator > 
"""

### File name
file_name = "Economically_active_2004_2022"
df = pd.read_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.csv", encoding = "utf8")

### Recode date columns
df.columns = [c.replace("Jan ", "01/") for c in df.columns]
df.columns = [c.replace("Dec ", "12/") for c in df.columns]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:21], value_name = "Value")\
    .rename(columns = {"variable": "Period"})
    
### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]
    
# Add "Variable Name" column
df_long["Variable Name"] = "Economic activity" 

# Add "Indicator" column
df_long["Indicator"] = "Economic activity" 

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Value"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")


##########################################
"""
   Economic unemployed
    
"""

### File name
file_name = "Economic_unemployment_2004_2022"
df = pd.read_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.csv", encoding = "utf8")

### Recode date columns
df.columns = [c.replace("Jan ", "01/") for c in df.columns]
df.columns = [c.replace("Dec ", "12/") for c in df.columns]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:21], value_name = "Value")\
    .rename(columns = {"variable": "Period"})

### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]
    
# Add "Variable Name" column
df_long["Variable Name"] = "Economic unemployment" 

# Add "Indicator" column
df_long["Indicator"] = "Economic unemployment" 

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Value"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")



##########################################
"""
   Economically Inactive
    
"""

### File name
file_name = "Economically_inactive_2004_2022"
df = pd.read_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Non ESS data/{file_name}.csv", encoding = "utf8")

### Recode date columns
df.columns = [c.replace("Jan ", "01/") for c in df.columns]
df.columns = [c.replace("Dec ", "12/") for c in df.columns]

### Create list of names
col_list = list(df.columns)

# Pivot wide to long
df_long = pd.melt(df, id_vars = ["AREACD", "AREANM"], value_vars = col_list[2:21], value_name = "Value")\
    .rename(columns = {"variable": "Period"})
    
### Apply Geomapper    
geomapper = load_geography_mapper(input_path, mapper_name)
df_long = apply_geography_mapper(df_long, geomapper)

# Drop Geomapper columns
df_long = df_long[["AREACD", "AREANM", "Geography", "Period", "Value"]]
    
# Add "Variable Name" column
df_long["Variable Name"] = "Economic inactivtiy" 

# Add "Indicator" column
df_long["Indicator"] = "Economic inactivity" 

# Add "Mission" column
df_long["Mission"] = "Mission 1"

# Add "Category" column
df_long["Category"] = "Spreading opportunity and improving public services"

# Add "Measure" column
df_long["Measure"] = "Value"

# Add "Unit" column
df_long["Unit"] = "Number"

# Restructure Columns
df_long = df_long[['AREACD', 'AREANM', 'Geography', 'Variable Name', 'Indicator',
                   'Mission', 'Category', 'Period', 'Value', 'Measure', 'Unit']]

### Save out data
df_long.to_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")



#####################################################################################################################

### File name
file_name = "Standard occupation Nomis data"
df = pd.read_csv(f"C:/Users/kellyj2/Office for National Statistics/Strategy, Coordination and Dissemination - Documents/Sharepoint_data_loading/JDAC/Cleaned data/{file_name}.csv")

bres_categories = df["Occupation Category"].unique()




