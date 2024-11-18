import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def parse_restaurant_data(input_folder: str, output_folder: str, total_pages: int = 100):
    """
    Parse restaurant data from HTML files and save the extracted information as TSV files.

    Parameters:
        input_folder (str): Path to the folder containing HTML files organized into subfolders (`page 1`, `page 2`, etc.).
        output_folder (str): Path to the folder where the parsed TSV files will be saved.
        total_pages (int, optional): Number of subfolders (pages) to process. Default is 100.

    Returns:
        None
    """
    def extract_price_and_cuisine(input_string):
        """
        Extract price range and cuisine type from the input string.

        Parameters:
            input_string (str): String containing restaurant attributes, such as price and cuisine.

        Returns:
            tuple: (price_range, cuisine_type)
        """
        # Extract price range (sequence of '€')
        price_range = re.search(r'€+', input_string)
        price_range = price_range.group() if price_range else ''
        
        # Extract cuisine type (after last '·')
        cuisine = input_string.split('·')[-1].strip()
        
        return price_range, cuisine

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    folders = list(range(1, total_pages + 1))
    rest_n = 1

    for fol in tqdm(folders, desc="Parsing folders"):
        # Get files in the current folder and order them based on restaurant numbers
        folder_path = os.path.join(input_folder, f'page {fol}')
        files = os.listdir(folder_path)
        files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

        for rest in files:
            # Load restaurant HTML
            with open(os.path.join(folder_path, rest), "r") as file:
                html = file.read()

            # Make HTML BeautifulSoup object
            html = BeautifulSoup(html, features='lxml')

            # Restaurant name
            restaurantName = html.select_one('h1', {'class' : 'data-sheet__title'}).text.strip()

            if restaurantName == '':
                print(f'restaurantName is missing in {rest}')

            # Restaurant location
            location = html.find_all('div', {'class' : 'data-sheet__block--text'})[0].text.strip()
            location = location.split(',')

            address = location[0]
            city = location[1]
            postalCode = location[2]
            country = location[3]
            
            if address == '':
                print(f'address is missing in {rest}')
            if city == '':
                print(f'city is missing in {rest}')
            if postalCode == '':
                print(f'postalCode is missing in {rest}')
            if country == '':
                print(f'country is missing in {rest}')
            
            # Restaurant characteristics
            attributes = html.find_all('div', {'class' : 'data-sheet__block--text'})[1].text.strip()
            # attributes = attributes.split()

            # priceRange = attributes[0]
            # cuisineType = attributes[-2] + attributes[-1]
            priceRange, cuisineType = extract_price_and_cuisine(attributes)

            if priceRange == '':
                print(f'priceRange is missing in {rest}')
            if cuisineType == '':
                print(f'cuisineType is missing in {rest}')


            # Restaurant desctiption
            if len(html.find_all('div', {'class' : 'data-sheet__description'})) != 0:
                description = html.find_all('div', {'class' : 'data-sheet__description'})[0].text.strip()
            else:
                print(f'description is missing in {rest}')
                description = ''
                

            # Restaurant services
            services = html.find_all('div', {'class' : 'col col-12 col-lg-6'})

            for ser in services:
                if len(ser.find_all('div')) == 0 :
                    facilitiesServices = [x.text.strip() for x in ser.find_all('li')]
                else:
                    creditCards = [x['data-src'].split('/')[-1].split('-')[0].capitalize() for x in ser.find_all('img')]
            
            if len(facilitiesServices) == 0:
                print(f'facilitiesServices is missing in {rest}')
            if len(creditCards) == 0:
                print(f'creditCards is missing in {rest}')

            # Restaurant phone number
            if len(html.find_all('a', {'data-event' : 'CTA_tel'})) !=0:
                phoneNumber = html.find_all('a', {'data-event' : 'CTA_tel'})[0]['href'].replace('tel:', '')
            else:
                print(f'phoneNumber is missing in {rest}')
                phoneNumber = ''

            # Restaurant website
            if len(html.find_all('a', {'data-event' : 'CTA_website'})) != 0:
                website = html.find_all('a', {'data-event' : 'CTA_website'})[0]['href']
            else:
                print(f'website is missing in {rest}')
                website = ''

            # Make dictionary
            restaurant_data = {
                    "restaurantName": restaurantName,                # string
                    "address": address,                              # string
                    "city": city,                                    # string
                    "postalCode": postalCode,                        # string
                    "country": country,                              # string
                    "priceRange": priceRange,                        # string
                    "cuisineType": cuisineType,                      # string
                    "description": description,                      # string
                    "facilitiesServices": f"{facilitiesServices}",        # list of strings, actually string (impossible to save list in tsv file)
                    "creditCards": f"{creditCards}",                      # list of strings, actually string (impossible to save list in tsv file)
                    "phoneNumber": phoneNumber,                      # string
                    "website": website                               # string
                }
            
            # List of restaurant data 
            data = [restaurant_data]

            # Save as TSV
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_folder, f"restaurant_{rest_n}.tsv"), sep="\t", index=False)
            rest_n += 1
 # type: ignore

