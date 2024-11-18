import folium
import pandas as pd
import webbrowser
import os
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

def geocode_with_postalcode(postal_code):
    
    geolocator = Nominatim(user_agent="my_geocoder")
    
    try:
        # Format the query to specifically look for Italian postal codes
        query = f"{postal_code}, Italia"
        
        # Use structured query with language set to Italian
        location = geolocator.geocode(
            query,
            language='it',
            addressdetails=True,  # This is key - it returns structured address details
            exactly_one=True
        )
        
        if location:
            address = location.raw['address']
            
            # Extract city (try different possible fields)
            city = (
                address.get('city') or
                address.get('town') or
                address.get('village') or
                address.get('municipality')
            )
            
            # Extract region/state
            region = (
                address.get('region') or
                address.get('state')
            )
            
            return {
                'postal_code': postal_code,
                'city': city,
                'region': region,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'full_address': address
            }
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error geocoding {postal_code}: {str(e)}")
        
    return None

def geocode_restaurants(df):
    
    # Create new columns
    df['latitude'] = None
    df['longitude'] = None
    df['city'] = None
    df['region'] = None
    
    # Cache for postal codes to avoid duplicate 
    postcode_cache = {}
    
    # Process each unique postal code
    for postal_code in df['postalCode'].unique():
        if pd.notna(postal_code):
            if postal_code not in postcode_cache:
                result = geocode_with_postalcode(str(postal_code))
                if result:
                    postcode_cache[postal_code] = result
                    time.sleep(1)  # Respect rate limits
    
    # Apply results to DataFrame
    for postal_code, data in postcode_cache.items():
        mask = df['postalCode'] == postal_code
        df.loc[mask, 'latitude'] = data['latitude']
        df.loc[mask, 'longitude'] = data['longitude']
        df.loc[mask, 'city'] = data['city']
        df.loc[mask, 'region'] = data['region']
    
    return df

