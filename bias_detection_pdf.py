import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
from reportlab.pdfgen import canvas
from fpdf import FPDF
import PyPDF2
from PIL import Image
import sys
import os
import io
from reportlab.lib.pagesizes import letter, landscape
import fitz
import datetime



logo = '/Users/nikoosabzevar/Desktop/Nikoo/synthesizeddata/test/Atb.jpg'

def main():
    script = sys.argv[0]
    filename = sys.argv[1]
    pred = sys.argv[2]
    train_data = pd.read_csv(filename)
    prediction = pred
    
    t1 = datetime.datetime.now()
    print(t1)
    
    print(prediction)

    print("dataset shape: ", train_data.shape)
    print("")

    features = train_data.columns.values
    print("Features are: ", end='')
    print(features)
    print("")

    unique_features = []
    for i in range(len(features)):
      if len(train_data[features[i]].unique()) == train_data.shape[0]:
        unique_features.append(features[i])
#

    features_categorical = [x for x in features if x not in train_data._get_numeric_data().columns]
    features_numerical = [x for x in features if x in train_data._get_numeric_data().columns]  
    
    for i in range(len(features_categorical)): 
        df = []
        if len(pd.to_numeric(train_data[features_categorical[i]], errors='coerce').dropna())!=0: 
            for j in range(len(train_data[features_categorical[i]])): 
                try: 
                    df.append(float(train_data[features_categorical[i]].values[j]))
                except ValueError: 
                    df.append(0)
            train_data[features_categorical[i]]=df

    discrete_value = [] #{}
    for i in range(len(features)):
      if len(train_data[features[i]].dropna().unique())<10:
#        print(features[i], end='')
#        print(train_data[features[i]].dropna().unique())
        discrete_value.append(features[i]+ ":"+str(train_data[features[i]].dropna().unique())) #[features[i]] = features[i]+ ":"+str(train_data[features[i]].dropna().unique())
#    print("")
    
#    msno.matrix(train_data)
      
    features_categorical = [x for x in features if x not in train_data._get_numeric_data().columns]
    features_numerical = [x for x in features if x in train_data._get_numeric_data().columns]


    t = train_data.duplicated(subset=None, keep ='first')
    if len(t[t==True])==0: 
      print("No Duplicate Rows")
      print("")
      
    # remove duplicates across the dataset 
    t = train_data.duplicated(subset=None, keep ='first') 
    t.index[t==True].tolist()
    for i in range(len(t[t==True])): 
      train_data.drop(t.index[t==True].tolist())
      
    print("Total Null or Nan cells per feature:")
    print(train_data.isna().sum())
    print("")
    
#    csv1 = {"Dataset Shape:":train_data.shape, "Features:": features, \
#            "Unique Features:": unique_features, "Summary of discrete features:": discrete_value, \
#            "Numerical Features:": features_numerical, "Categorical Features:": features_categorical}
    

############## Summary Statistics ###################
    pdf=FPDF(format='letter', unit='in')
    pdf.add_page()
    pdf.set_font('Times','',10.0)
    pdf.cell(7.8,10.2, border=1)
    pdf.ln(0.05)
    pdfdir = filename[0:filename.rfind('/')+1]+'Report0.pdf'
    pdf.output(pdfdir,'F')       
    ########
    pages_to_keep = [1] # page numbering starts from 0
    infile = PdfFileReader(pdfdir, 'rb')
    output = PdfFileWriter()
    
    for i in range(infile.getNumPages()):
        p = infile.getPage(i)
        if i in pages_to_keep: # getContents is None if page is blank
            output.addPage(p)
    with open(pdfdir, 'wb') as f:
       output.write(f)
    ###########################################
    c = canvas.Canvas(pdfdir)
    c.drawImage(logo, 200, 760, 200, 110)
    c.showPage()
    c.save()
    ##################
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont('Times-Bold', 15)
    can.drawString(190, 760, "DATA TRUST - BIAS DETECTION")

    can.setFont('Times-Bold', 13)    
    can.drawString(50, 720, "Summary of Features/Attributes")

    can.setFont('Times-Roman', 11)

    can.drawString(50, 670, "Number of rows: " + str(train_data.shape[0])+ ', Number of columns: '+ str(train_data.shape[1]))        

    if len(" ".join(str(x) for x in features)) // 100 + min(1, len(" ".join(str(x) for x in features))% 100)==0:
        i3=0
    else: 
        for i3 in range(len(" ".join(str(x) for x in features)) // 100 + min(1, len(" ".join(str(x) for x in features))% 100)): 
            t = " ".join(str(x) for x in features)[i3*100:(i3+1)*100]
            can.drawString(50, 630-i3*10, t)

    if len(" ".join(str(x) for x in unique_features)) // 100 + min(1, len(" ".join(str(x) for x in unique_features))% 100)==0: 
        j = 0
    else: 
        for j in range(len(" ".join(str(x) for x in unique_features)) // 100 + min(1, len(" ".join(str(x) for x in unique_features))% 100)): 
            can.drawString(50, 590-i3*10-j*10, " ".join(str(x) for x in unique_features).replace('\n','')[j*100:(j+1)*100])
 
    if len(" ".join(str(x) for x in features_categorical)) // 100 + min(1, len(" ".join(str(x) for x in features_categorical))% 100)==0:
        i= 0
    else: 
        for i in range(len(" ".join(str(x) for x in features_categorical)) // 100 + min(1, len(" ".join(str(x) for x in features_categorical))% 100)): 
            can.drawString(50, 550-i3*10-j*10-i*10, " ".join(str(x) for x in features_categorical).replace('\n','')[i*100:(i+1)*100])

    if len(" ".join(str(x) for x in features_numerical)) // 100 + min(1, len(" ".join(str(x) for x in features_numerical))% 100) ==0: 
        i1 =0
    else: 
        for i1 in range(len(" ".join(str(x) for x in features_numerical)) // 100 + min(1, len(" ".join(str(x) for x in features_numerical))% 100)): 
            can.drawString(50, 510-i3*10-j*10-i*10-i1*10, " ".join(str(x) for x in features_numerical).replace('\n','')[i1*100:(i1+1)*100])
    
    if len(" ".join(str(x) for x in discrete_value)) // 100 + min(1, len(" ".join(str(x) for x in discrete_value))% 100) ==0:
        i4 = 0
    else: 
        for i4 in range(len(" ".join(str(x) for x in discrete_value)) // 100 + min(1, len(" ".join(str(x) for x in discrete_value))% 100)): 
            can.drawString(50, 470-i3*10-j*10-i*10-i1*10-i4*10, " ".join(str(x) for x in discrete_value).replace('\n','')[i4*100:(i4+1)*100])
   
    can.setFont('Times-Bold', 11) 
    can.drawString(50, 680, "Database Shape:")    
    can.drawString(50, 640, "List of Features:")
    can.drawString(50, 600-i3*10, "Unique Features (They may represent sensitive/confidential information):")
    can.drawString(50, 560-i3*10-j*10, "List of Categorical Features:")
    can.drawString(50, 520-i3*10-j*10-i*10, "List of Numerical Features:")
    can.drawString(50, 480-i3*10-j*10-i*10-i1*10, "List of Discrete Features:")
    
    
    can.save()
    
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    with open(pdfdir,"rb") as f:
        existing_pdf = PyPDF2.PdfFileReader(f)
    #existing_pdf = PdfFileReader(file(dir1, "rb"))
        output = PdfFileWriter()
        page = existing_pdf.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)
        dir0 = pdfdir[0:pdfdir.rfind('/')+1]+'Report1.pdf'
        with open(dir0, "wb") as f2:
            output.write(f2)


    fig, ax = plt.subplots()
    sns.heatmap(train_data.corr(method='pearson'), annot=True, fmt='.4f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.savefig(filename[0:filename.rfind('/')+1]+'corr.png', bbox_inches='tight', pad_inches=0.0)
    
             
    doc = fitz.open(dir0)          # open the PDF
    rect = fitz.Rect(50, 830-250-i3*10-j*10-i*10-i1*10-i4*10, 450, 1100-250-i3*10-j*10-i*10-i1*10-i4*10)
    
    for page in doc:
        page.insertImage(rect, filename[0:filename.rfind('/')+1]+'corr.png')
    
    doc.saveIncr()    
    
    ########################
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont('Times-Bold', 11)
    
    corr = train_data.corr()
    cor_threshold = 0.7

    t = np.asarray(corr)
    if np.asarray(np.where(abs(t)==1)).shape[1]==int(np.sqrt(corr.size)):
        can.drawString(250, 50, "No strong Correlation") 

    else: 
        for i in range(len(corr.columns.values)):
          ind = np.where(abs(t[i])>cor_threshold)
          for j in range(len(ind[0])):
            if int(ind[0][j])>i: 
                can.drawString(100, 50-i*10, "There is some level of correlation "+ str(round(t[i, ind[0][j]], 3))+" between " +  corr.columns.values[i] + " and "+ corr.columns.values[ind[0][j]])
              
    can.save()
    
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    with open(dir0,"rb") as f:
        existing_pdf = PyPDF2.PdfFileReader(f)
    #existing_pdf = PdfFileReader(file(dir1, "rb"))
        output = PdfFileWriter()
        page = existing_pdf.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)
        dir1 = pdfdir[0:pdfdir.rfind('/')+1]+'Report2.pdf'
        with open(dir1, "wb") as f2:
            output.write(f2)
###################################################
        # PAGE 2

    index = []
    skewed_feat = {}
    for i in range(len(features_numerical)):  
      # remove Nan/Null and duplicate for all features 
      feature = train_data[features_numerical[i]].dropna() #train_data.iloc[:, i].dropna()
      #outlier_datapoints_zmodify = detect_outlier_zmodified(feature)

      #detect skewness
      skewness = feature.skew(axis = 0, skipna = True)

      
      skewness_threshold = 0.7
      if abs(skewness)>skewness_threshold: # or 100*len(detect_outlier_zmodified(feature))/len(feature)>40:
        print("===SKEWNESS WARNING===" + " Skewness = "+ str(skewness))
        print(train_data[features_numerical[i]].describe())
        print("Total Nan or Nulls: " + str(train_data[features_numerical[i]].isna().sum()))
        print("=======================================")
        index.append(i)
        
      skewed_feat[features_numerical[i]] = skewness
    
    t = [features_numerical[x] for x in index]
    f_cat = [x for x in t if x not in unique_features if x not in prediction]
    df = train_data[f_cat]
    fig, ax = plt.subplots(max(1, len(index)//3+1*len(index)%3), min(3, len(index)), sharey=True, tight_layout=True)
    if index==[]:
      print("No skewness detected in the numerical features' distributions")
    else:
    
      fig, ax = plt.subplots(max(1, len(f_cat)//5+1*min(1, len(f_cat)%5)), min(5, len(f_cat))) 
      if len(f_cat)==1: 
        for i, f_catt in enumerate(df):         
          df[f_catt].value_counts().plot("bar", ax=ax).set_title(f_catt)
#        for axi in ax.flat: 
#            axi.xaxis.set_major_locator(plt.MaxNLocator(3))  
      else:           
        if len(f_cat)//5==0 or (len(f_cat)==5):
          for i, f_catt in enumerate(df):
              df[f_catt].value_counts().plot("bar", ax=ax[i]).set_title(f_catt)
          for axi in ax.flat: 
              axi.xaxis.set_major_locator(plt.MaxNLocator(3))     
        else: 
          for i, f_catt in enumerate(df):
            df[f_catt].value_counts().plot("bar", ax=ax[i//5, i%5]).set_title(f_catt)
            ax[i//5, i%5].xaxis.set_major_locator(plt.MaxNLocator(3))

    plt.savefig(filename[0:filename.rfind('/')+1]+'skewness_numerical.png', bbox_inches='tight', pad_inches=0.0)

# Categorical Features

    index_categorical = []
    ind_ax = []
    for j in range(len(features_categorical)):  
      # remove Nan/Null and duplicate for all features 
      feature = train_data[features_categorical[j]].dropna() #train_data.iloc[:, i].dropna()
      pdf_category = train_data[features_categorical[j]].value_counts()/train_data[features_categorical[j]].count()
      
      if len(pdf_category)>5:
        ind_ax.append(features_categorical[j])
     
      #detect skewness
      if features_categorical[j] not in unique_features:
        
        pdf = np.sort(pdf_category)
        skewness = 0
        for i in range(len(pdf)-1):
          if pdf[i+1]/pdf[i]-1 > skewness: 
            skewness = pdf[i+1]/pdf[i]-1 
      
        skewed_feat[features_categorical[j]] = skewness
      
        skewness_threshold = 0.7
        if abs(skewness)>skewness_threshold: # or 100*len(detect_outlier_zmodified(feature))/len(feature)>40:
          print("===SKEWNESS WARNING===" + " Skewness = "+ str(skewness))
          print(train_data[features_categorical[j]].describe())
          print("Total Nan or Nulls: " + str(train_data[features_categorical[j]].isna().sum()))
          print("=======================================")
          index_categorical.append(j)


    t = [features_categorical[x] for x in index_categorical]
    f_cat = [x for x in t if x not in unique_features if x not in prediction]

    df = train_data[f_cat]
    fig, ax = plt.subplots(max(1, len(f_cat)//5+1*min(1, len(f_cat)%5)), min(5, len(f_cat)))

    for i, f_catt in enumerate(df):
      if len(f_cat)//5==0:
        for i, f_catt in enumerate(df):
            df[f_catt].value_counts().plot("bar", ax=ax[i]).set_title(f_catt)
        inn = 0
        for axi in ax.flat: 
          if f_cat[inn] in ind_ax:
            axi.xaxis.set_major_locator(plt.MaxNLocator(3))
          inn += 1 
      else:    
        df[f_catt].value_counts().plot("bar", ax=ax[i//5, i%5]).set_title(f_catt)
        ax[i//5, i%5].xaxis.set_major_locator(plt.MaxNLocator(3))

    fig.show()
    plt.savefig( filename[0:filename.rfind('/')+1]+'skewness_categorical.png', bbox_inches='tight', pad_inches=0.0)
        

    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont('Times-Bold', 15)
    can.drawString(190, 760, "DATA TRUST - BIAS DETECTION")

    can.setFont('Times-Bold', 13)    
    can.drawString(50, 720, "Skewed Attributes")

    can.setFont('Times-Roman', 11)

    can.drawString(50, 670, "Skewed Numerical Attributes")        

    can.drawString(50, 350, "Skewed Categorical Attributes")        
    
    can.save()
    
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    with open(pdfdir,"rb") as f:
        existing_pdf = PyPDF2.PdfFileReader(f)
    #existing_pdf = PdfFileReader(file(dir1, "rb"))
        output = PdfFileWriter()
        page = existing_pdf.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)
        dir2 = pdfdir[0:pdfdir.rfind('/')+1]+'Report3.pdf'
        with open(dir2, "wb") as f2:
            output.write(f2)


    doc = fitz.open(dir2)          # open the PDF
    rect = fitz.Rect(50, 180, 400, 480)
    
    for page in doc:
        page.insertImage(rect, filename[0:filename.rfind('/')+1]+'skewness_numerical.png')
    
    doc.saveIncr()   
    ######
    doc = fitz.open(dir2)          # open the PDF
    rect = fitz.Rect(50, 510, 400, 810)
    
    for page in doc:
        page.insertImage(rect, filename[0:filename.rfind('/')+1]+'skewness_categorical.png')
    
    doc.saveIncr()    


########################
    # PAGE 3
    # Assessing Fairness
    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont('Times-Bold', 15)
    can.drawString(190, 760, "DATA TRUST - BIAS DETECTION")

    can.setFont('Times-Bold', 13)    
    can.drawString(50, 720, "Assessing Fairness")

    can.setFont('Times-Bold', 12)

    can.drawString(50, 700, "Disparate Impact") 
    can.drawString(50, 400, "Predictive Parity") 
       
    can.setFont('Times-Roman', 12)

    can.drawString(50, 680, "Dataset has potential DISPARATE IMPACT (80% RULE) with respect to: ")               
    can.drawString(50, 380, "Dataset lacks Predictive PARITY if the ratio conditional probability is significantly greater than 1")
    


    #Y = train_data[prediction]
    feat_fair = []
    for i in range(len(features_categorical)):
      if (features_categorical[i] not in prediction) and (i in index_categorical) and len(train_data[features_categorical[i]].dropna().unique())<5:
        feat_fair.append(features_categorical[i])
        
    pre_out = train_data[prediction].dropna().unique()
    for i in range(len(feat_fair)):
      print("")
      df = train_data[[prediction, feat_fair[i]]].dropna()
      thsh = 1
      for i1 in range(len(train_data[feat_fair[i]].dropna().unique())-1):
        for i2 in range(i1, len(train_data[feat_fair[i]].dropna().unique())):
          t1 = len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i1])])/len(df[df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]])
          t2 = len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i2])])/len(df[df[feat_fair[i]]==df[feat_fair[i]].unique()[i2]])
          if thsh > (min(t1, t2)/max(t1, t2)):
            thsh = (min(t1, t2)/max(t1, t2))
            ind = [min(t1, t2), max(t1, t2), [t1, t2].index(min(t1, t2)), [t1, t2].index(max(t1, t2))]

      can.setFont('Times-Roman', 11)
      can.drawString(50, 660-20*i, "P("+prediction+"|"+feat_fair[i]+"="+df[feat_fair[i]].unique()[ind[2]]+") = "+str(round(ind[0], 2))+ " vs. P("+prediction+"|"+feat_fair[i]+"="+df[feat_fair[i]].unique()[ind[3]]+") = "+str(round(ind[1], 2)))

  
    can.save()
    
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    with open(pdfdir,"rb") as f:
        existing_pdf = PyPDF2.PdfFileReader(f)
    #existing_pdf = PdfFileReader(file(dir1, "rb"))
        output = PdfFileWriter()
        page = existing_pdf.getPage(0)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)
        dir3 = pdfdir[0:pdfdir.rfind('/')+1]+'Report4.pdf'
        with open(dir3, "wb") as f2:
            output.write(f2)
    


    #predic_parity = {}
    equal_matx = np.zeros((len(feat_fair), len(feat_fair))) + 1
    for i in range(len(feat_fair)):
      for j in range(len(feat_fair)):
        if i!=j: 
          df = train_data[[prediction, feat_fair[i], feat_fair[j]]].dropna()
          mn = 1
          mx = 0
     
          for i1 in range(len(train_data[feat_fair[i]].dropna().unique())):
            if len(df[(df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])==0: 
              mn = mx = 1
              pass
            else: 
              if len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])/len(df[(df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])<mn:
                mn = len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])/len(df[(df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])
              if len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])/len(df[(df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])>mx:
                mx = len(df[(df[prediction]==pre_out[0]) & (df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])/len(df[(df[feat_fair[i]]==df[feat_fair[i]].unique()[i1]) & (df[feat_fair[j]]==df[feat_fair[j]].unique()[0])])

          equal_matx[i,j] = mx/mn

    
    df = pd.DataFrame(equal_matx, columns = [feat_fair], index = [feat_fair])
    

    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, fmt='.4f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.savefig(filename[0:filename.rfind('/')+1]+'Predictive_Parity.png', bbox_inches='tight', pad_inches=0.0)


    doc = fitz.open(dir3)          # open the PDF
    rect = fitz.Rect(50, 500, 450, 750)
    
    for page in doc:
        page.insertImage(rect, filename[0:filename.rfind('/')+1]+'Predictive_Parity.png')
    
    doc.saveIncr()    
    
    
    merger = PdfFileMerger()
    all_files = [dir1, dir2, dir3]
    for f in all_files: 
        merger.append(PdfFileReader(f), 'b')

    final_report = pdfdir[0:pdfdir.rfind('/')+1]+'BiasDetectionReport.pdf'
    merger.write(final_report)

    t2 = datetime.datetime.now()
    
    print(t2)
        
if __name__ == '__main__':
   main()
