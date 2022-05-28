import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from ipywidgets import FloatSlider,IntSlider,interact
from sklearn.preprocessing import LabelEncoder

sn.set()



data = pd.read_csv("cleaneddata.csv")
df=data
le = LabelEncoder()
label = le.fit_transform(df["Type"])
df = df.drop("Type",axis ="columns" )
df['Type']=label
pages = st.sidebar.selectbox('Select an option',
										options = ['Data Analysis',
													'Recommendation Engine'])
le = LabelEncoder()
label = le.fit_transform(df["Child_Safety_Locks"])
df = df.drop("Child_Safety_Locks",axis ="columns" )
df['Child_Safety_Locks']=label

	
st.title("Car Recommendation & Analysis Project")
container1= st.container()

def RE(CSL,P,M,SC,FTC,BS,X):
	print(P,M,X)
	test = df[["Price","ARAI_Certified_Mileage","Seating_Capacity","Fuel_Tank_Capacity","Boot_Space","Child_Safety_Locks","Type"]]
	test.fillna(0, inplace = True)
	# Create K-Nearest Neighbors
	print(test.head())
	nn = NearestNeighbors(n_neighbors=3).fit(test.values)
	print(nn.kneighbors([[P,M,SC,FTC,BS,CSL,X]]))
	res=df.iloc[nn.kneighbors([[P,M,SC,FTC,BS,CSL,X]])[1][0]]
	print(res)
	return res
			

if pages=="Recommendation Engine":
	container1.subheader("Recommendation Engine")
	form1=st.form(key = "my_form")
	form1.write("What kind of Car do you want?")
	p=form1.slider("Price",min_value=200000,max_value=1000000)
	m=form1.slider("Mileage",10,50)
	sc=form1.number_input("Seating Capacity",2,7)
	ftc=form1.slider("Fuel Tank Capacity",10,80)
	bs=form1.slider("Boot Space",20,100)
	tt=form1.selectbox('Transmission Type',
												options=["Manual","Automatic","DCT","CVT","AMT"])
	csl=form1.checkbox("Child Safety Locks", False)									
	submitted = form1.form_submit_button("Recommend")
			
	if submitted:
		x = {
			"Manual":0,"Automatic":1,"DCT":2,"CVT":3,"AMT":4
			}
		t=RE((1 if csl else 0),p,m,sc,ftc,bs,x[tt])
		print(t)
		with st.expander(label="Here is your Recommendation"):
			i=1
			for _,x in t.iterrows():
				with st.container():
					st.subheader(str(i)+") "+str(x.Make)+" "+str(x.Model))
					st.text("Price: Rs. "+str(x["Price"]))
					st.text("Body Type: "+str(x["Body_Type"]))
					st.text("Mileage: "+str(x["ARAI_Certified_Mileage"]))
					st.text("Fuel Tank Capacity: "+str(x.Fuel_Tank_Capacity) +" litres")
					st.text("Seating capacity: "+str(x.Seating_Capacity))
					i+=1
		

	
	
     
	
if pages=="Data Analysis":
	st.sidebar.checkbox("Show analysis by car features", True, key = 1)
	st.title("Analysis Dashboard")
	st.markdown("The dashboard will help a researcher to get to know \
	more about the given datasets and it's output")
	st.sidebar.title("Select Visual Charts")
	st.sidebar.markdown("Select the Plots accordingly:")
	selected_status = st.sidebar.selectbox('Car features',
										options = ['Make',
													'Model', 'Price',
													'ARAI_Certified_Mileage','Seating_Capacity',
													'Front_Brakes', 'Fuel_Tank_Capacity',
													'Body_Type','Child_Safety_Locks',
													'Boot_Space',
													'Type','Fuel_Type'])
												
	fig = go.Figure()
				
	def plotMake():
		st.title("Make analysis")
		fig = plt.figure(figsize=(25,10))
		plt.bar(data['Make'].value_counts().keys(),data['Make'].value_counts().values)
		st.pyplot(fig)



	def plotModel():
		st.title("Model analysis")
		fig = plt.figure(figsize=(30,5))
		plt.bar(data['Model'].value_counts().keys(),data['Model'].value_counts().values)
		plt.xticks(rotation=90)
		st.pyplot(fig)
		

	def plotPrice():
		dict=data['Price'].value_counts(bins=10,sort=False)
		print(dict.keys(),dict.values)
		st.title("Price analysis")
		fig = plt.figure(figsize=(25,10))
		sn.barplot(dict.index, dict.values, alpha=1)
		st.pyplot(fig)

	def plotARAI_Certified_Mileage():
		st.title("ARAI_Certified_Mileage analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['ARAI_Certified_Mileage'].value_counts().keys(),data['ARAI_Certified_Mileage'].value_counts().values)
		st.pyplot(fig)

	def plotSeating_Capacity():
		st.title("Seating_Capacity Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Seating_Capacity'].value_counts().keys(),data['Seating_Capacity'].value_counts().values)
		st.pyplot(fig)


	def plotFront_Brakes():
		st.title("Front_Brakes Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Front_Brakes'].value_counts().keys(),data['Front_Brakes'].value_counts().values)
		st.pyplot(fig)


	def plotFuel_Tank_Capacity():
		st.title("Fuel_Tank_Capacity Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Fuel_Tank_Capacity'].value_counts().keys(),data['Fuel_Tank_Capacity'].value_counts().values)
		st.pyplot(fig)

	def plotBody_Type():
		st.title("Body_Type Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Body_Type'].value_counts().keys(),data['Body_Type'].value_counts().values)
		st.pyplot(fig)

	def plotChild_Safety_Locks():
		st.title("Child_Safety_Locks")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Child_Safety_Locks'].value_counts().keys(),data['Child_Safety_Locks'].value_counts().values)
		plt.xticks(rotation=90)
		st.pyplot(fig)

	def plotBoot_Space():
		st.title("Boot_Space Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Boot_Space'].value_counts().keys(),data['Boot_Space'].value_counts().values)
		plt.xticks(rotation=90)
		st.pyplot(fig)




	def plotType():
		st.title("Transmission Type Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Type'].value_counts().keys(),data['Type'].value_counts().values)
		plt.xticks(rotation=90)
		st.pyplot(fig)

	def plotFuel_Type():
		st.title("Fuel_Type Analysis")
		fig = plt.figure(figsize=(25,5))
		plt.bar(data['Fuel_Type'].value_counts().keys(),data['Fuel_Type'].value_counts().values)
		plt.xticks(rotation=90)
		st.pyplot(fig)

		
		
		
	if selected_status == 'Make':
			plotMake()
	if selected_status == 'Model':
			plotModel()	
	if selected_status == 'Price':
			plotPrice()							
	if selected_status == 'ARAI_Certified_Mileage':
			plotARAI_Certified_Mileage()
	if selected_status == 'Seating_Capacity':
			plotSeating_Capacity()
	if selected_status == 'Front_Brakes':
			plotFront_Brakes()
	if selected_status == 'Fuel_Tank_Capacity':
			plotFuel_Tank_Capacity()		
	if selected_status == 'Body_Type': 
			plotBody_Type()
	if selected_status == 'Child_Safety_Locks': 
			plotChild_Safety_Locks()
	if selected_status == 'Boot_Space': 
			plotBoot_Space()
	if selected_status == 'Type': 
			plotType()
	if selected_status == 'Fuel_Type': 
			plotFuel_Type()


	