import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import re


#extract testing feature
def extract_feature_test(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain
#extract training feature
def extract_feature_train(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain



#main funcation
def main():
    excel_file= 'training.xlsx'
    df=pd.DataFrame(pd.read_excel(excel_file))

    excel_file_test= 'test1.xlsx'
    df1=pd.DataFrame(pd.read_excel(excel_file_test))

    a=[]
    b=[]
    a1=[]
    b1=[]
    for url in df['url']:
        a.append(url)

    for output in df['phishing']:
        b.append(output)

    for url1 in df1['url']:
        a1.append(url1)

    for output in df1['result']:
        b1.append(output)


    c=[]
    d=[]
    for url1,output1 in zip(a,b):
       
        url=url1
        output=output1
        c.append(extract_feature_train(url,output))
    

    for url1,output1 in zip(a1,b1):
           
        url=url1
        output=output1
        d.append(extract_feature_test(url,output))
    


    df=pd.DataFrame(c,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df.to_csv('feature_train.csv', sep=',', encoding='utf-8')

    df_test=pd.DataFrame(d,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df_test.to_csv('feature_test.csv', sep=',', encoding='utf-8')

    print('DONE')



  
if __name__== "__main__":
  main()





    
