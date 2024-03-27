###################### RECYCLING POLICIES FRONTEND #################################
# Recycling policies by city
# Lyon
lyon_RP_message = "\nBrown bin : organic \nGreen bin : glass \nBlue bin : cardboard, paper, plastic, metal \nGrey bin : trash"
lyon_bins = {
    'trash': 'grey',
    'glass': 'green',
    'paper': 'blue',
    'cardboard': 'blue',
    'plastic': 'blue',
    'metal': 'blue',
    'organic' : 'brown'
    }
lyon_bins_images = {
    'trash': 'lyon_grey_bin.png',
    'glass': 'lyon_green_bin.png',
    'paper': 'lyon_blue_bin.png',
    'cardboard': 'lyon_blue_bin.png',
    'plastic': 'lyon_blue_bin.png',
    'metal': 'lyon_blue_bin.png',
    'organic' : 'lyon_brown_bin.png'
    }

# Other city
other_city_RP_message = "\nBrown bin : organic \nGreen bin : glass \nBlue bin : cardboard, paper, plastic, metal \nGrey bin : trash"
other_city_bins = {
    'trash': 'grey',
    'glass': 'green',
    'paper': 'blue',
    'cardboard': 'blue',
    'plastic': 'blue',
    'metal': 'blue',
    'organic' : 'brown'
    }
other_city_bins_images = {
    'trash': 'lyon_grey_bin.png',
    'glass': 'lyon_green_bin.png',
    'paper': 'lyon_blue_bin.png',
    'cardboard': 'lyon_blue_bin.png',
    'plastic': 'lyon_blue_bin.png',
    'metal': 'lyon_blue_bin.png',
    'organic' : 'lyon_brown_bin.png'
    }

# Recycling points by city (paths to databases)
# Lyon
lyon_recycling_points_files = {
    'trash': 'lyon_grey.csv',
    'glass': 'lyon_green.csv',
    'paper': 'lyon_blue.csv',
    'cardboard': 'lyon_blue.csv',
    'plastic': 'lyon_blue.csv',
    'metal': 'lyon_blue.csv',
    'organic' : 'lyon_brown.csv'
    }

# Custom policies [policy text, bin-class correspondance, images of the bins, data files for recycling points]
custom_policies = {
    'Lyon' : [lyon_RP_message, lyon_bins, lyon_bins_images, lyon_recycling_points_files],
    'Other city': [other_city_RP_message, other_city_bins, other_city_bins_images, None]
    }
