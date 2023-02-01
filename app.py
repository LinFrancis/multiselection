import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


#__________________________________________________________________________________________________________________________________________________________________
# Dashboard structure
#__________________________________________________________________________________________________________________________________________________________________
st.set_page_config(page_title="Explorer", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

CURRENT_THEME = "light"
IS_DARK_THEME = False

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

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
#__________________________
#General Information Survey
#__________________________
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

#__________________________
#Pledge Statement Survey
#__________________________
df_pdg = pd.read_csv('Pledge_27012013.csv',sep=';', header=None, prefix="g").iloc[2:]
df_pdg.set_index("g0", inplace = True)
df_pdg.index.names = ['Master ID']
df_pdg = df_pdg.dropna(how = 'all')
df_pdg_names = pd.read_csv('Pledge_27012013.csv',sep=';').iloc[1:]
##rename
replacement_mapping_dict = {"Not targeted.": "0","2": "1","Mildly targeted but not exclusively.": "2","4": "3",
    "Main or unique target.": "4",}
df_pdg['g28'] = df_pdg['g28'].replace(['Poor'], 'Low income communities')
df_pdg["g20"] = df_pdg["g20"].replace(replacement_mapping_dict) #Women and girls ind
df_pdg["g21"] = df_pdg["g21"].replace(replacement_mapping_dict) #LGBTQIA+ ind
df_pdg["g22"] = df_pdg["g22"].replace(replacement_mapping_dict) #Elderly ind
df_pdg["g23"] = df_pdg["g23"].replace(replacement_mapping_dict) #Children and Youth ind
df_pdg["g24"] = df_pdg["g24"].replace(replacement_mapping_dict) #Disabled ind
df_pdg["g25"] = df_pdg["g25"].replace(replacement_mapping_dict) #Indigenous or traditional communities ind
df_pdg["g26"] = df_pdg["g26"].replace(replacement_mapping_dict) #Racial, ethnic and/or religious minorities ind
df_pdg["g27"] = df_pdg["g27"].replace(replacement_mapping_dict) #Refugees ind
df_pdg["g28"] = df_pdg["g28"].replace(replacement_mapping_dict) #Low income communities ind

##change format
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
##Creating list of series needed to treat multiquestionsº.
hazard_list_ind         = df_pdg.iloc[:, 32:50].apply(lambda x: x.str.strip()).columns.values.tolist() #Individual Engagement
hazard_list_comp        = df_pdg.iloc[:, 460:478].apply(lambda x: x.str.strip()).columns.values.tolist() #Companies engagement
hazard_list_countries   = df_pdg.iloc[:, 867:885].apply(lambda x: x.str.strip()).columns.values.tolist() #Countries
hazard_list_region      = df_pdg.iloc[:, 1274:1292].apply(lambda x: x.str.strip()).columns.values.tolist() #Regions
hazard_list_cities      = df_pdg.iloc[:, 1681:1699].apply(lambda x: x.str.strip()).columns.values.tolist() #Cities
hazard_list_nat_sys     = df_pdg.iloc[:, 2095:2113].apply(lambda x: x.str.strip()).columns.values.tolist() #Natural Systems
hazards_options         = ['Heat stress - lives & livelihoods combined','Heat stress - livelihoods (work)','Heat stress - lives','Extreme heat','Extreme cold','Snow and ice','Drought (agriculture focus)','Drought (other sectors)','Water stress (urban focus)','Water stress (rural focus)','Fire weather (risk of wildfires)','Urban flooding','Riverine flooding','Coastal flooding','Other coastal events','Oceanic events','Hurricanes/cyclones','Extreme wind']
companies_type_list     = df_pdg.iloc[:, 436:457].apply(lambda x: x.str.strip()).columns.values.tolist()
nat_syst_type_list      = df_pdg.iloc[:, 2079:2089].apply(lambda x: x.str.strip()).columns.values.tolist()


#All Hazards Treatment
df_pdg[hazard_list_ind] = df_pdg[hazard_list_ind].where(df_pdg['g32'] != 'All Hazard', hazards_options)  #Recode "All Hazard" = Apply to all Hazard"
df_pdg[hazard_list_comp] = df_pdg[hazard_list_comp].where(df_pdg['g460'] != 'All Hazard', hazards_options)
df_pdg[hazard_list_countries] = df_pdg[hazard_list_countries].where(df_pdg['g867'] != 'All Hazard', hazards_options)
df_pdg[hazard_list_region] = df_pdg[hazard_list_region].where(df_pdg['g1274'] != 'All Hazard', hazards_options)
df_pdg[hazard_list_cities] = df_pdg[hazard_list_cities].where(df_pdg['g1681'] != 'All Hazard', hazards_options)
df_pdg[hazard_list_nat_sys] = df_pdg[hazard_list_nat_sys].where(df_pdg['g2095'] != 'All Hazard', hazards_options)
#Concatenating columns
df_pdg['Engagement scope'] = df_pdg[['g13','g14','g15','g16','g17','g18']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Hazards_ind'] = df_pdg[hazard_list_ind].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1) #Concatenate
df_pdg['Hazards_comp'] = df_pdg[hazard_list_comp].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Hazards_countries'] = df_pdg[hazard_list_countries].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Hazards_region'] = df_pdg[hazard_list_region].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Hazards_cities'] = df_pdg[hazard_list_cities].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Hazards_nat_sys'] = df_pdg[hazard_list_nat_sys].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['All_Hazards'] = df_pdg[['Hazards_ind','Hazards_comp','Hazards_countries','Hazards_region','Hazards_cities','Hazards_nat_sys']].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Companies Types'] = df_pdg[companies_type_list].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
df_pdg['Natural Systems Types'] = df_pdg[nat_syst_type_list].apply(lambda x:'; '.join(x.dropna().astype(str)),axis=1)
#Confirmation that works!
    #st.write(hazard_list_ind,hazard_list_comp, hazard_list_countries,hazard_list_region,hazard_list_cities,hazard_list_nat_sys)
    #st.table(df_pdg[['g32','Hazards_ind','g460','Hazards_comp','g867','Hazards_countries','g1274','Hazards_region','g1681','Hazards_cities','g2095','Hazards_nat_sys','All_Hazards']])
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

#confirmar uso de dummy item para vincular encuesta plan. Ver video aquí: https://www.youtube.com/watch?v=iZUH1qlgnys&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE&index=7


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
    ['Engagement scope',['Individuals','Companies','Countries','Regions','Cities','Natural Systems','']],   #extend the tables cats_defs, cats, defs, poss if needed
    ['All Hazards',['Heat stress - lives & livelihoods combined','Heat stress - livelihoods (work)','Heat stress - lives','Extreme heat','Extreme cold','Snow and ice','Drought (agriculture focus)','Drought (other sectors)','Water stress (urban focus)','Water stress (rural focus)','Fire weather (risk of wildfires)','Urban flooding','Riverine flooding','Coastal flooding','Other coastal events','Oceanic events','Hurricanes/cyclones','Extreme wind','']] ]  #extend the tables cats_defs, cats, de

cats = [cats_defs[0][0], cats_defs[1][0]     , cats_defs[2][0]  ,cats_defs[3][0],cats_defs[4][0]       ]  #list of question categories
defs = [cats_defs[0][1], cats_defs[1][1]     , cats_defs[2][1]  ,cats_defs[3][1],cats_defs[4][1]      ]  #list of possible answers
poss = [df['Region']   , df['Priority group'], df['Impact System'] ,df['Engagement scope'],df['All_Hazards']]  #correspoding answers

regions_options = ['Oceania & Pacific','East Asia','South Asia','East Europe & Central Asia','Northern & Western Europe','North Africa and the Middle East','Sub-Saharan Africa','South America','Central America and Caribbean','North America']
priority_options = ['Women and girls','LGBTQIA+ people','Elderly','Children & Youth','Indigenous and traditional communities','Ethnic or religious minorities','Refugees','Disabled People','Low income communities']
areas_options = ['Cross-cutting enablers: Planning and Finance','Food and Agriculture Systems','Coastal and Oceanic Systems','Water and Nature Systems','Human Settlement Systems','Infrastructure Systems']
engagement_options = ['Individuals','Companies','Countries','Regions','Cities','Natural Systems']
hazards_options = ['Heat stress - lives & livelihoods combined','Heat stress - livelihoods (work)','Heat stress - lives','Extreme heat','Extreme cold','Snow and ice','Drought (agriculture focus)','Drought (other sectors)','Water stress (urban focus)','Water stress (rural focus)','Fire weather (risk of wildfires)','Urban flooding','Riverine flooding','Coastal flooding','Other coastal events','Oceanic events','Hurricanes/cyclones','Extreme wind']

st.sidebar.image("R2R_RGB_PINK.png")
st.sidebar.markdown("SELECT R2R PARTNER INFORMATION")
st.sidebar.caption("If no specific filter is selected, all available information regarding R2R partners will be displayed. Please select filters for a more targeted display.")
#st.sidebar.markdown('**Resiliencia**')
areas_selection = st.sidebar.multiselect("Partner's Impact Systems",      areas_options)
engagement_selection = st.sidebar.multiselect("Partner's Engagement scope", engagement_options)  #add further multiselect if needed
#st.sidebar.markdown('**Vulnerablilidad**')
priority_selection = st.sidebar.multiselect('Priority groups aimed to make more resilient',   priority_options)
#st.sidebar.markdown('**Amenazas**')
hazards_selection = st.sidebar.multiselect('Hazards to provide resilience',   hazards_options)
#st.sidebar.markdown('**Territorialidad**')
macro_region_selection = st.sidebar.multiselect('Macro Regions where they operate',    regions_options)

selection = [macro_region_selection,priority_selection,areas_selection,engagement_selection,hazards_selection]   #extend if more multiselect

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
selection_all = [["Oceania & Pacific","East Asia","South Asia","East Europe & Central Asia","Northern & Western Europe","North Africa and the Middle East","Sub-Saharan Africa","South America","Central America and Caribbean","North America",""],
  ["Women and girls","LGBTQIA+ people","Elderly","Children & Youth","Indigenous and traditional communities","Ethnic or religious minorities","Refugees","Disabled People","Low income communities",""],
  ["Cross-cutting enablers: Planning and Finance","Food and Agriculture Systems","Coastal and Oceanic Systems","Water and Nature Systems","Human Settlement Systems","Infrastructure Systems",""],
  ["Individuals","Companies","Countries","Regions","Cities","Natural Systems",""],
  ["Heat stress - lives & livelihoods combined","Heat stress - livelihoods (work)","Heat stress - lives","Extreme heat","Extreme cold","Snow and ice","Drought (agriculture focus)","Drought (other sectors)","Water stress (urban focus)","Water stress (rural focus)","Fire weather (risk of wildfires)","Urban flooding","Riverine flooding","Coastal flooding","Other coastal events","Oceanic events","Hurricanes/cyclones","Extreme wind",""]]

st.header('R2R DATA EXPLORER V2.0 Trial Fase')
if selection == selection_all:
    st.subheader('**COMPLETE R2R PARTNERS INFORMATION ON DISPLAY**')
else:
    st.subheader('DISPLAYED RESULTS SHOW R2R PARTNERS INFORMATION MEETING ALL SELECTED CRITERIA')

col1,col2,col3,col4 = st.columns((1,1,1,3))
col1.caption('Original dataframe shape')
col1.write(df.shape)
col2.caption('Filtered dataframe shape')
col2.write(df_filtered.shape)
with st.expander("Review Raw Data"):
        st.write(df_filtered[['Initiative_name','Short name','Priority group','Impact System','Engagement scope','Region','All_Hazards']])
st.markdown("""---""")
#__________________________________________________________________________________________________________________________________________________________________
# HAZARDS  PLEDGE
#__________________________________________________________________________________________________________________________________________________________________
#Sample size hazards from selection.
df2_sz = df_filtered['All_Hazards'].replace(['; ; ; ; ; '], np.nan).dropna()
sz = str(df2_sz.count())
#dataframe to workwith (All data from the selection).
df2 = df_filtered['All_Hazards'].str.split(";", expand=True).apply(lambda x: x.str.strip())
df2 = df2.stack().value_counts()
df2 = df2.iloc[1:].sort_index().reset_index(name='Frecuency')  #Check what happend whit blank information.
#creating new columms based in a dictionary
heat_list     = {'Heat stress - lives & livelihoods combined':'Heat','Heat stress - livelihoods (work)':'Heat','Heat stress - lives':'Heat','Extreme heat':'Heat'}
cold_list     = {'Extreme cold':'Cold','Snow and ice':'Cold'}
drought_list  = {'Drought (agriculture focus)':'Drought','Drought (other sectors)':'Drought'}
water_list    = {'Water stress (urban focus)':'Water','Water stress (rural focus)':'Water'}
fire_list     = {'Fire weather (risk of wildfires)':'Fire'}
flooding_list = {'Urban flooding':'Flooding','Riverine flooding':'Flooding','Coastal flooding':'Flooding'}
coastal_list  = {'Other coastal events':'Coastal / Ocean','Oceanic events':'Coastal / Ocean'}
wind_list     = {'Hurricanes/cyclones':'Wind','Extreme wind':'Wind'}
hazard_dictionary = {**heat_list,**cold_list,**drought_list,**water_list,**fire_list,**flooding_list,**coastal_list,**wind_list}
df2['group'] = df2['index'].map(hazard_dictionary)
df2['% hazard'] = ((df2['Frecuency']/df2['Frecuency'].sum())*100).round(1)
#df2 = df2.groupby(['group']).value_counts(normalize=True).sort_index().reset_index(name='%group')
#df2['%group'] = df2['%index']*100

#treemap
st.write(df2)
fig = px.treemap(df2, path=[px.Constant("Hazards"),'group','index'], values = '% hazard')
fig.update_traces(root_color="lightgray")
fig.update_layout(title_text='Hazards aimed to provide resilience by R2R Partners')
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
if selection == selection_all:
    fig.add_annotation(x=1, y=0,
                text='Out of '+ sz +' Partners reporting information related to Harzards (Source: All Data. R2R Pledge Attributes Survey)',showarrow=False,
                yshift=-20)
else:
    fig.add_annotation(x=1, y=0,
                text='Out of '+ sz +' Partners reporting information related to Harzards (Source: Data filtered by users selection. R2R Pledge Attributes Survey)',showarrow=False,
                yshift=-20)
st.plotly_chart(fig)
st.markdown("""---""")
#__________________________________________________________________________________________________________________________________________________________________
# PRIORITY GROUPS PLEDGE
#__________________________________________________________________________________________________________________________________________________________________
#
#Sample size hazards from selection.
df2_sz = df_filtered['Priority group'].replace(['; ; ; ; ; '], np.nan).dropna()
sz = str(df2_sz.count())
#dataframe to workwith (All data from the selection).
df2 = df_filtered
list = {'g20','g21','g22','g23','g24','g25','g26','g27','g28'} #making a list with all the columns name use in the graph
df2= df2[df2[list].notna()] #cleaning na
pg0 = df2["g20"].mean() #Women and girls
pg1 = df2["g21"].mean() #LGBTQIA+
pg2 = df2["g22"].mean() #Elderly
pg3 = df2["g23"].mean() #Children and Youth
pg4 = df2["g24"].mean() #Disabled
pg5 = df2["g25"].mean() #Indigenous or traditional communities
pg6 = df2["g26"].mean() #Racial, ethnic and/or religious minorities
pg7 = df2["g27"].mean() #Refugees
pg8 = df2["g28"].mean() #Low income communities

s_df2 = pd.DataFrame(dict(
    r=[pg0, pg1, pg2, pg3, pg4, pg5, pg6, pg7, pg8],
    theta=['Women and girls','LGBTQIA+','Elderly','Children and Youth','Disabled','Indigenous or traditional communities','Racial, ethnic and/or religious minorities','Refugees','Low income communities']))
s_fig_ra_general = px.line_polar(s_df2, r='r', theta='theta', line_close=True, title="Priority groups (Only for Individuals Scope)")
s_fig_ra_general.update_traces(line_color='#FF37D5', line_width=1)
s_fig_ra_general.update_traces(fill='toself')
s_fig_ra_general.add_annotation(x=1, y=0,
            text='Out of '+ sz +' Partners reporting information about the Priority Groups pledged to make more resilient',showarrow=False,
            yshift=-60)
st.write(s_fig_ra_general)
st.markdown("""---""")
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
st.markdown("Scatterplot for coastal/rural in individual scope (Mean of % of all Engagement Scope)")

x = df2['C']
y = df2['R']
z = df2['Name']

fig = plt.figure(figsize=(10, 10))

for i in range(len(df2)):
    plt.scatter(x,y,c='#FF37D5', marker='o')

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
st.markdown("""---""")
#__________________________________________________________________________________________________________________________________________________________________
# Companies type
#__________________________________________________________________________________________________________________________________________________________________
#
df2 = df_filtered['Companies Types'].str.split(";", expand=True).apply(lambda x: x.str.strip())
df2 = df2.stack().value_counts()
df2 = df2.iloc[1:].sort_index().reset_index(name='Frecuency')
df2['Percentaje'] = ((df2['Frecuency']/df2['Frecuency'].sum())*100).round(2)

companies_type_rename_list = {'A. Agriculture, forestry and fishing':'Agriculture, forestry and fishing','B. Mining and quarrying':'Mining and quarrying','C. Manufacturing':'Manufacturing','D. Electricity, gas, steam and air conditioning supply':'Electricity, gas, steam and air conditioning supply','E. Water supply; sewerage, waste management and remediation activities':'Water supply; sewerage, waste management and remediation activities','F. Construction':'Construction','G. Wholesale and retail trade; repair of motor vehicles and motorcycles':'Wholesale and retail trade; repair of motor vehicles and motorcycles','H. Transportation and storage':'Transportation and storage','I. Accommodation and food service activities':'Accommodation and food service activities','J. Information and communication':'Information and communication','K. Financial and insurance activities':'Financial and insurance activities','L. Real estate activities':'Real estate activities','M. Professional, scientific and technical activities':'Professional, scientific and technical activities','N. Administrative and support service activities':'Administrative and support service activities','O. Public administration and defence; compulsory social security':'Public administration and defence; compulsory social security','P. Education':'Education','Q. Human health and social work activities':'Human health and social work activities','R. Arts, entertainment and recreation':'Arts, entertainment and recreation','S. Other service activities':'Other service activities','T. Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use':'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use','U. Activities of extraterritorial organizations and bodies':'Activities of extraterritorial organizations and bodies'}
df2['index'] = df2['index'].replace(companies_type_rename_list)

fig, ax = plt.subplots()
ax  = sns.barplot(x="Percentaje", y="index", data=df2,label="Types of Companies", color="#FF37D5")
ax.bar_label(ax.containers[0],padding=3)
#ax.set_xlim(right=15)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set(ylabel=None)
plt.title('Types of companies as R2R Partners Members', fontsize=13, loc='left')
st.pyplot(fig)
st.markdown("""---""")

#__________________________________________________________________________________________________________________________________________________________________
# Natural Systems
#__________________________________________________________________________________________________________________________________________________________________

df2 = df_filtered['Natural Systems Types'].str.split(";", expand=True).apply(lambda x: x.str.strip())
df2 = df2.stack().value_counts()
df2 = df2.iloc[1:].sort_index().reset_index(name='Frecuency')
df2['Percentaje'] = ((df2['Frecuency']/df2['Frecuency'].sum())*100).round(1)

nat_sys_dict ={'T Terrestrial':'Terrestrial','S Subterranean':'Subterranean','SF Subterranean-Freshwater':'Subterranean-Freshwater','SM Subterranean-Marine':'Subterranean-Marine','FT Freshwater-Terrestrial':'Freshwater-Terrestrial','F Freshwater':'Freshwater','FM Freshwater-Marine':'Freshwater-Marine','M Marine':'Marine','MT Marine-Terrestrial':'Marine-Terrestrial','MFT Marine-Freshwater-Terrestrial':'Marine-Freshwater-Terrestrial'}
df2['index'] = df2['index'].replace(nat_sys_dict)

fig, ax = plt.subplots()
ax  = sns.barplot(x="Percentaje", y="index", data=df2,label="Types of Natural Systems", color="#FF37D5")
ax.bar_label(ax.containers[0],padding=3)
#ax.set_xlim(right=70)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set(ylabel=None)
plt.title('Types of Natural Systems pledged to have an impact', fontsize=13, loc='left')
st.pyplot(fig)
st.markdown("""---""")
