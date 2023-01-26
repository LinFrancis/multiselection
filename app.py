import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
#import re

#__________________________________________________________________________________________________________________________________________________________________
# Dashboard structure
#__________________________________________________________________________________________________________________________________________________________________
st.set_page_config(page_title="Explorer", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

# Hide index when showing a table. CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
#__________________________________________________________________________________________________________________________________________________________________
# Export data
#__________________________________________________________________________________________________________________________________________________________________

#General Information Survey
df_gi = pd.read_csv('GI_27012013.csv',sep=';', header=None, prefix="q").iloc[2:]
df_gi.set_index("q0", inplace = True)
df_gi.index.names = ['Master ID']
df_gi = df_gi.dropna(how = 'all')
df_gi_names = pd.read_csv('GI_27012013.csv',sep=';').iloc[1:]

##rename
df_gi['q37'] = df_gi['q37'].replace(['Poor'], 'Low income communities')
df_gi['q50'] = df_gi['q50'].replace(['Finance'], 'Cross-cutting enablers: Planning and Finance')
df_gi['q51'] = df_gi['q51'].replace(['Food and agriculture system'], 'Food and Agriculture Systems')
df_gi['q52'] = df_gi['q52'].replace(['Ocean and coastal zone'], 'Coastal and Oceanic Systems')
df_gi['q53'] = df_gi['q53'].replace(['Water and land ecosystems'], 'Water and Nature Systems')
df_gi['q54'] = df_gi['q54'].replace(['Cities and human settlements'], 'Human Settlement Systems')
df_gi['q55'] = df_gi['q55'].replace(['Infrastructure and services'], 'Infrastructure Systems')
#creating new variables concatenating
df_gi['Region']           = df_gi[['q39','q40','q41','q42','q43','q44','q45','q46','q47','q48']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_gi['Priority group']   = df_gi[['q29','q30','q31','q32','q33','q34','q35','q36','q37',     ]].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_gi['Impact System']       = df_gi[['q50','q51','q52','q53','q54','q55'                        ]].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#Pledge Statement Survey
df_pdg = pd.read_csv('Pledge_27012013.csv',sep=';', header=None, prefix="g").iloc[2:]
df_pdg.set_index("g0", inplace = True)
df_pdg.index.names = ['Master ID']
df_pdg = df_pdg.dropna(how = 'all')
df_pdg_names = pd.read_csv('Pledge_27012013.csv',sep=';').iloc[1:]

##rename
df_pdg['g29'] = df_pdg['g29'].replace(['Poor'], 'Low income communities')
#creating new variables concatenating
df_pdg['Engagement scope'] = df_pdg[['g13','g14','g15','g16','g17','g18']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#hazards for indivdual engagement
table_hazards = df_pdg.iloc[:, 33:51] #selecting all columns and making a new dataframe
v_hazards = table_hazards.columns.values.tolist() #making a list with the names of the columns
df_pdg['Hazards'] = df_pdg[v_hazards].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#continents for indivdual engagement
table_continents_ind = df_pdg.iloc[:, 52:57] #selecting all columns and making a new dataframe
v_continents_ind = table_continents_ind.columns.values.tolist() #making a list with the names of the columns
df_pdg['continents_ind'] = df_pdg[v_continents_ind].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#countries for indivdual engagement
table_countries_ind = df_pdg.iloc[:, 58:243] #selecting all columns and making a new dataframe
v_countries_ind = table_countries_ind.columns.values.tolist() #making a list with the names of the columns
df_pdg['countries_ind'] = df_pdg[v_countries_ind].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#Plan Statement Survey
df_plan = pd.read_csv('Plan_27012013.csv',sep=';', header=None, prefix="p").iloc[2:]
df_plan.set_index("p0", inplace = True)
df_plan.index.names = ['Master ID']
df_plan = df_plan.dropna(how = 'all')

#Resilience Attributes Survey
df_ra = pd.read_csv('RA_27012013.csv',sep=';', header=None, prefix="r").iloc[2:]
df_ra.set_index("r0", inplace = True)
df_ra.index.names = ['Master ID']
df_ra = df_ra.dropna(how = 'all')

#Making one database
df = pd.concat([df_gi,df_pdg,df_ra], axis=1)
df_len = len(df.index)
#st.write(df.shape)

#__________________________________________________________________________________________________________________________________________________________________
# MULTISELECTOR
#__________________________________________________________________________________________________________________________________________________________________
#

cats_defs = [
    ['Region',['Oceania & Pacific','East Asia','South Asia','East Europe & Central Asia','Northern & Western Europe','North Africa and the Middle East','Sub-Saharan Africa','South America','Central America and Caribbean','North America','']],
    ['Priority group',  ['Women and girls','LGBTQIA+ people','Elderly','Children & Youth','Indigenous and traditional communities','Ethnic or religious minorities','Refugees','Disabled People','Low income communities','']],
    ['Impact System',      ['Cross-cutting enablers: Planning and Finance','Food and Agriculture Systems','Coastal and Oceanic Systems','Water and Nature Systems','Human Settlement Systems','Infrastructure Systems','']],
    ['Engagement scope',['Individuals','Companies','Countries','Regions','Cities','Natural Systems','']]  ] #extend the tables cats_defs, cats, defs, poss if needed

cats = [cats_defs[0][0], cats_defs[1][0]     , cats_defs[2][0]  ,cats_defs[3][0]       ]  #list of question categories
defs = [cats_defs[0][1], cats_defs[1][1]     , cats_defs[2][1]  ,cats_defs[3][1]       ]  #list of possible answers
poss = [df['Region']   , df['Priority group'], df['Impact System'] ,df['Engagement scope']]  #correspoding answers

regions_options = ['Oceania & Pacific','East Asia','South Asia','East Europe & Central Asia','Northern & Western Europe','North Africa and the Middle East','Sub-Saharan Africa','South America','Central America and Caribbean','North America']
priority_options = ['Women and girls','LGBTQIA+ people','Elderly','Children & Youth','Indigenous and traditional communities','Ethnic or religious minorities','Refugees','Disabled People','Low income communities']
areas_options = ['Cross-cutting enablers: Planning and Finance','Food and Agriculture Systems','Coastal and Oceanic Systems','Water and Nature Systems','Human Settlement Systems','Infrastructure Systems']
engagement_options = ['Individuals','Companies','Countries','Regions','Cities','Natural Systems']

st.sidebar.write('Buscador')
macro_region_selection = st.sidebar.multiselect('Macro Region where they operate',    regions_options)
priority_selection = st.sidebar.multiselect('Priority group which describes them',   priority_options)
areas_selection = st.sidebar.multiselect('Impact Systems where they operate',      areas_options)
engagement_selection = st.sidebar.multiselect('Their engagement scope', engagement_options)   #add further multiselect if needed

selection = [macro_region_selection,priority_selection,areas_selection,engagement_selection]           #extend if more multiselect
#selection = [[],[],[],['Natural Systems']]                        #if the selection is empty, all items to be considered (handling empty strimlit multiselections)
#st.write(selection)
#st.write(df['Engagement scope'])

i=0
while i < len(selection):
    if len(selection[i])==0:
        selection[i]=defs[i]
    i=i+1

def index_selection_results(sel,col):
        results_index = []
        i=0
        while i < df_len:                  #going over all the rows
            for elem in sel:               #going over all the items in the selection
                if elem in col[i]:         #checking if item is contained in the string
                    results_index.append(i) #saving the correct item fulfilling the selection
            i=i+1
        return results_index

def common_member(a, b):                   #used to intersect any two lists
    result = [i for i in a if i in b]
    return result

final_list = list(range(0,df_len+1))
j = 0
while j < len(selection)-1:
        temp_list  = list(set(index_selection_results(selection[j],poss[j]))) #avoidung index duplications
        final_list = list(set(common_member(temp_list,final_list)))
        j = j+1

df_filtered = df.iloc[final_list].reset_index().sort_values(by = 'q2')
df_filtered.set_index("Master ID", inplace = True)

st.markdown('Resultados')
col1,col2,col3,col4 = st.columns((1,1,1,3))
col1.caption('Original dataframe shape')
col1.write(df.shape)
col2.caption('Filtered dataframe shape')
col2.write(df_filtered.shape)
col3.caption('Selection vector')
col3.write(final_list)
st.markdown("Problema: el buscador no selecciona información del 4 selector: df['Engagement scope']. Todo lo demás funciona bien.")
st.write(df_filtered[['Region','Priority group','Impact System','Engagement scope']])
#st.write(df_filtered[['Engagement scope']])
