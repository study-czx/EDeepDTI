from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import warnings



def get_famlily_info_from_KEGG(th_element, df_lis):
    td_element = th_element.find_next('td')
    a_elements = td_element.find_all('a')
    for family in a_elements:
        this_dict = {'Uniprot': Uniprot_id, 'KEGG': KEGG_id, 'family': family.text}
        record = pd.DataFrame.from_dict(this_dict, orient='index').T
        if df_lis.empty:
            df_lis = record
        else:
            df_lis = pd.concat([df_lis, record])
    return df_lis



warnings.filterwarnings('ignore')
proxies = {"http": None, "https": None}  # Disable proxy

datasets = ['DTI', 'CPI']

ids_without_family = []

for dataset in datasets:
    Uniprot_KEGG = pd.read_excel('KEGG/' + dataset + '/Uniprot_KEGG.xlsx', header=0, sheet_name='Sheet0')
    # print(Uniprot_KEGG)

    count_without_family = 0

    lis = pd.DataFrame()
    # lis_text = pd.DataFrame()

    for i in range(len(Uniprot_KEGG)):
        Uniprot_id = Uniprot_KEGG['From'][i]
        KEGG_id = Uniprot_KEGG['To'][i]
        print('Uniprot ID: {}, KEGG ID: {}'.format(Uniprot_id, KEGG_id))

        url = "https://www.kegg.jp/entry/" + KEGG_id

        resp = requests.get(url, proxies=proxies, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        th_element = soup.find('th', string='Brite')

        if th_element:
            lis = get_famlily_info_from_KEGG(th_element, lis)

        else:
            flag = 0
            for k in range(10):
                print('try again: round ', k)
                time.sleep(3)
                try:
                    resp = requests.get(url, proxies=proxies, timeout=10)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    th_element = soup.find('th', string='Brite')
                    if th_element:
                        lis = get_famlily_info_from_KEGG(th_element, lis)
                        break
                except:
                    continue

            if flag == 0:
                print("未找到 Brite 相关信息")
                ids_without_family.append([Uniprot_id, KEGG_id])


    print('没有Family信息的蛋白数量')
    print(len(ids_without_family))
    print(ids_without_family)
    df_ids_without_family = pd.DataFrame(ids_without_family, columns=['Uniprot_id', 'KEGG'])

    lis.to_csv('KEGG/' + dataset + '/protein_family_info.csv', index=False)
    df_ids_without_family.to_csv('KEGG/' + dataset + '/protein_without_family_info.csv', index=False)
