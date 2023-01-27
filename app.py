import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import mplcursors
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
#______
#General Information Survey
#______
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
df_gi.rename(columns = { 'q1':'Initiative_name', 'q2':'Short name'}, inplace = True)
#creating new variables concatenating
df_gi['Region']           = df_gi[['q39','q40','q41','q42','q43','q44','q45','q46','q47','q48']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_gi['Priority group']   = df_gi[['q29','q30','q31','q32','q33','q34','q35','q36','q37',     ]].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_gi['Impact System']    = df_gi[['q50','q51','q52','q53','q54','q55'                        ]].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#____
#Pledge Statement Survey
#______
df_pdg = pd.read_csv('Pledge_27012013.csv',sep=';', header=None, prefix="g").iloc[2:]
df_pdg.set_index("g0", inplace = True)
df_pdg.index.names = ['Master ID']
df_pdg = df_pdg.dropna(how = 'all')
df_pdg_names = pd.read_csv('Pledge_27012013.csv',sep=';').iloc[1:]
##rename
df_pdg['g28'] = df_pdg['g28'].replace(['Poor'], 'Low income communities')

replacement_mapping_dict = {"Not targeted.": "0","2": "1","Mildly targeted but not exclusively.": "2","4": "3",
    "Main or unique target.": "4",}

df_pdg["g20"] = df_pdg["g20"].replace(replacement_mapping_dict) #Women and girls ind
df_pdg["g21"] = df_pdg["g21"].replace(replacement_mapping_dict) #LGBTQIA+ ind
df_pdg["g22"] = df_pdg["g22"].replace(replacement_mapping_dict) #Elderly ind
df_pdg["g23"] = df_pdg["g23"].replace(replacement_mapping_dict) #Children and Youth ind
df_pdg["g24"] = df_pdg["g24"].replace(replacement_mapping_dict) #Disabled ind
df_pdg["g25"] = df_pdg["g25"].replace(replacement_mapping_dict) #Indigenous or traditional communities ind
df_pdg["g26"] = df_pdg["g26"].replace(replacement_mapping_dict) #Racial, ethnic and/or religious minorities ind
df_pdg["g27"] = df_pdg["g27"].replace(replacement_mapping_dict) #Refugees ind
df_pdg["g28"] = df_pdg["g28"].replace(replacement_mapping_dict) #Low income communities ind

df_pdg["g20"] = pd.to_numeric(df_pdg["g20"]) #Women and girls ind
df_pdg["g21"] = pd.to_numeric(df_pdg["g21"]) #LGBTQIA+ ind
df_pdg["g22"] = pd.to_numeric(df_pdg["g22"]) #Elderly ind
df_pdg["g23"] = pd.to_numeric(df_pdg["g23"]) #Children and Youth ind
df_pdg["g24"] = pd.to_numeric(df_pdg["g24"]) #Disabled ind
df_pdg["g25"] = pd.to_numeric(df_pdg["g25"]) #Indigenous or traditional communities ind
df_pdg["g26"] = pd.to_numeric(df_pdg["g26"]) #Racial, ethnic and/or religious minorities ind
df_pdg["g27"] = pd.to_numeric(df_pdg["g27"]) #Refugees ind
df_pdg["g28"] = pd.to_numeric(df_pdg["g28"]) #Low income communities ind
df_pdg["g30"] = pd.to_numeric(df_pdg["g30"])
df_pdg["g31"] = pd.to_numeric(df_pdg["g31"])
df_pdg["g458"] = pd.to_numeric(df_pdg["g458"])
df_pdg["g459"] = pd.to_numeric(df_pdg["g459"])
df_pdg["g865"] = pd.to_numeric(df_pdg["g865"])
df_pdg["g1272"] = pd.to_numeric(df_pdg["g1272"])
df_pdg["g1273"] = pd.to_numeric(df_pdg["g1273"])
df_pdg["g1679"] = pd.to_numeric(df_pdg["g1679"])
df_pdg["g1680"] = pd.to_numeric(df_pdg["g1680"])
df_pdg["g2093"] = pd.to_numeric(df_pdg["g2093"])
df_pdg["g2094"] = pd.to_numeric(df_pdg["g2094"])


#creating new variables concatenating
df_pdg['Engagement scope'] = df_pdg[['g13','g14','g15','g16','g17','g18']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#hazards for indivdual engagement
#table_hazards = df_pdg.iloc[:, 33:51] #selecting all columns and making a new dataframe
#v_hazards = table_hazards.columns.values.tolist() #making a list with the names of the columns
#df_pdg['Hazards'] = df_pdg[v_hazards].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#continents for indivdual engagement
#table_continents_ind = df_pdg.iloc[:, 52:57] #selecting all columns and making a new dataframe
#v_continents_ind = table_continents_ind.columns.values.tolist() #making a list with the names of the columns
#df_pdg['continents_ind'] = df_pdg[v_continents_ind].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

#countries for indivdual engagement
#table_countries_ind = df_pdg.iloc[:, 58:243] #selecting all columns and making a new dataframe
#v_countries_ind = table_countries_ind.columns.values.tolist() #making a list with the names of the columns
#df_pdg['countries_ind'] = df_pdg[v_countries_ind].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)

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
areas_selection = st.sidebar.multiselect('Impact Systems',      areas_options)
engagement_selection = st.sidebar.multiselect('Engagement scope', engagement_options)   #add further multiselect if needed
priority_selection = st.sidebar.multiselect('Priority groups',   priority_options)
macro_region_selection = st.sidebar.multiselect('Macro Regions',    regions_options)

selection = [macro_region_selection,priority_selection,areas_selection,engagement_selection]           #extend if more multiselect

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
while j < len(selection):
        temp_list  = list(set(index_selection_results(selection[j],poss[j]))) #avoidung index duplications
        final_list = list(set(common_member(temp_list,final_list)))
        j = j+1

df_filtered = df.iloc[final_list].reset_index().sort_values(by = 'Short name')
df_filtered.set_index("Master ID", inplace = True)

#__________________________________________________________________________________________________________________________________________________________________
# MAIN RESULTS
#__________________________________________________________________________________________________________________________________________________________________
#

st.markdown('Resultados')
col1,col2,col3,col4 = st.columns((1,1,1,3))
col1.caption('Original dataframe shape')
col1.write(df.shape)
col2.caption('Filtered dataframe shape')
col2.write(df_filtered.shape)
st.write(df_filtered[['Initiative_name','Short name','Priority group','Impact System','Engagement scope']])


#__________________________________________________________________________________________________________________________________________________________________
# PRIORITY GROUPS PLEDGE
#__________________________________________________________________________________________________________________________________________________________________
#

df2 = df_filtered

list = {'g20','g21','g22','g23','g24','g25','g26','g27','g28'} #making a list with all the columns name use in the graph

df2= df2[df2[list].notna()] #cleaning na

pg0 = df_pdg["g20"].mean() #Women and girls
pg1 = df_pdg["g21"].mean() #LGBTQIA+
pg2 = df_pdg["g22"].mean() #Elderly
pg3 = df_pdg["g23"].mean() #Children and Youth
pg4 = df_pdg["g24"].mean() #Disabled
pg5 = df_pdg["g25"].mean() #Indigenous or traditional communities
pg6 = df_pdg["g26"].mean() #Racial, ethnic and/or religious minorities
pg7 = df_pdg["g27"].mean() #Refugees
pg8 = df_pdg["g28"].mean() #Low income communities

s_df2 = pd.DataFrame(dict(
    r=[pg0, pg1, pg2, pg3, pg4, pg5, pg6, pg7, pg8],
    theta=['Women and girls','LGBTQIA+','Elderly','Children and Youth','Disabled','Indigenous or traditional communities','Racial, ethnic and/or religious minorities','Refugees','Low income communities']))
s_fig_ra_general = px.line_polar(s_df2, r='r', theta='theta', line_close=True, title="Vulnerable groups (Only for Individuals Scope)")
s_fig_ra_general.update_traces(fill='toself')

st.write(s_fig_ra_general)


#__________________________________________________________________________________________________________________________________________________________________
# SCATTER PLOT INLAND. RURAL - All Engagement Scope
#__________________________________________________________________________________________________________________________________________________________________
#
#Data preparation
df2 = df_filtered

costal_list = {'g30','g458','g865','g1272','g1679','g2093'}
rural_list  = {'g31','g459','g866','g1273','g1680','g2094'}

df2_costal  = df2[costal_list]
df2_rural   = df2[rural_list ]

df2['costal_average'] = df2_costal.mean(axis=1,numeric_only=True,skipna=True)
df2['rural_average'] = df2_rural.mean(axis=1,numeric_only=True,skipna=True)

df2 = df2[df2['costal_average'].notna()]
df2 = df2[df2['rural_average'].notna()]

df2.rename(columns = {'costal_average':'C', 'rural_average':'R', 'Short name':'Name'}, inplace = True)

#Scatterplot for coastal/rural in individual scope

x = df2['C']
y = df2['R']
z = df2['Name']

fig = plt.figure(figsize=(10, 10))
#placeholder = st.empty()

for i in range(len(df2)):
    plt.scatter(x,y,c='blue', marker='o')
plt.title("Scatterplot for coastal/rural (Mean of % of all Engagement Scope)",fontsize=14)
#plt.title("Individuals' environment ""[%]""",fontsize=14)
plt.xlabel('Inland'+' '*74+'Coastal',fontsize=13)
plt.ylabel('Urban'+' '*49+'Rural',fontsize=13)

plt.gca().spines['top']  .set_visible(False)
plt.gca().spines['right'].set_visible(False)

for i in range(len(df2)):
     plt.text(df2.C[df2.Name ==z[i]],df2.R[df2.Name==z[i]],z[i], fontdict=dict(color='black', alpha=0.5, size=12))

plt.xlim([0, 100])
plt.ylim([0, 100])    #more ideas: https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_scatter.html

col1, col2, col3 = st.columns((0.4,2.2,0.4))
col2.pyplot(fig)


#__________________________________________________________________________________________________________________________________________________________________
# SCATTER PLOT INLAND. RURAL
#__________________________________________________________________________________________________________________________________________________________________
#
#Data preparation
#df2 = df_filtered

#df2= df2[df2['g30'].notna()] #Individual coastal
#df2= df2[df2['g31' ].notna()] #Individual rural

#df2['g30'] = df2['g30'].astype(int) #Individual coastal
#df2['g31']   = df2['g31'  ].astype(int) #Individual rural

#df2.rename(columns = {'g30':'C', 'g31':'R', 'Short name':"Name"}, inplace = True)

#Scatterplot for coastal/rural in individual scope
#st.markdown("Scatterplot for coastal/rural in individual scope (Only for Individuals Scope)")

#x = df2['C']
#y = df2['R']
#z = df2['Name']

#fig = plt.figure(figsize=(10, 10))
#placeholder = st.empty()
#for i in range(len(df2)):
#    plt.scatter(x,y,c='red', marker='o')

#plt.title("Individuals' environment ""[%]""",fontsize=13)
#plt.xlabel('inland'+' '*74+'coastal',fontsize=11)
#plt.ylabel('urban'+' '*49+'rural',fontsize=11)

#plt.gca().spines['top']  .set_visible(False)
#plt.gca().spines['right'].set_visible(False)

#for i in range(len(df2)):
#     plt.text(df2.C[df2.Name ==z[i]],df2.R[df2.Name==z[i]],z[i], fontdict=dict(color='black', alpha=0.5, size=12))

#plt.xlim([0, 100])
#plt.ylim([0, 100])    #more ideas: https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_scatter.html

#placeholder.pyplot(fig)
