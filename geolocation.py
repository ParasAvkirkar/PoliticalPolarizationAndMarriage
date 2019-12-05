#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import geopy
import pandas as pd


# In[ ]:


couples_df = pd.read_csv('couples/20140319/couples_FLA_20140319.csv', sep='\t')


# In[ ]:


couples_df.head()


# In[ ]:


couples_df['uniq_addr'] = couples_df[['residence_addr_line_1_L', 'residence_city_L']].apply(lambda x: ' '.join(x), axis=1)


# In[ ]:


addr_to_zip = {couples_df.uniq_addr[i]: couples_df.residence_zipcode_5_L[i] for i in range(len(couples_df))}


# In[66]:


uniq_addresses = couples_df['uniq_addr'].unique()


# In[67]:


len(uniq_addresses)


# In[68]:


uniq_addresses


# In[63]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="pp")
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geocode({ 'postalcode': '32164'})


# In[64]:


location


# In[ ]:


from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import tqdm
geolocator = Nominatim(user_agent="pp")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.8)

failed = []
with open('geolocation.csv', 'w') as file:
    for addr in tqdm.tqdm(uniq_addresses):
#         if i % 100 == 0:
#             print('Iteration: ' + str(i))
        try:
            location = geocode(addr)
        except exception as e:
            location = None
        if location:
            write_line = '\t'.join([addr, str(location.address), str(location.latitude), str(location.longitude), str(location.altitude)]) + '\n'
            file.write(write_line)
        else:
            try:
                location = geocode({'postalcode': addr_to_zip[addr]})
            except Exception as e:
                location = None
            if location:
                write_line = '\t'.join([addr, str(location.address), str(location.latitude), str(location.longitude), str(location.altitude)]) + '\n'
                file.write(write_line)
            else:
                failed.append(addr)
#                 print('Failed: ' + str(len(failed)))


# In[ ]:




