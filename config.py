RENAME_MAP           = {
                        'Name': 'name',
                        'Pronouns': 'pronouns',
                        'UFL Email': 'ufl_email',
                        'Phone': 'phone',
                        'Socials': 'socials',
                        'Year': 'year',
                        'Major': 'major',
                        'Other Orgs': 'other_orgs',
                        'Role (0=Big,1=Little)': 'role',
                        'Preferred Littles': 'preferred_littles',
                        'Preferred Bigs': 'preferred_bigs',
                        'Pairing Requests (Optional)': 'pairing_requests',
                        'On/Off Campus (0=On,1=Off)': 'on_off_campus',
                        'Has Car (0=No,1=Yes)': 'has_car',
                        'Ideal Big/Little': 'ideal_big_little',
                        'Looking For ACE': 'looking_for_ace',
                        'Free Time': 'free_time',
                        'Hobbies': 'hobbies',
                        'Favorite Artists/Songs': 'favorite_artists_songs',
                        'Icks': 'dislikes',
                        'Talk for Hours About': 'talk_for_hours_about',
                        'Self Description': 'self_description',
                        'Best Joke': 'best_joke',
                        'Favorite Food': 'favorite_food',
                        'EarlyBird/NightOwl (0=Early,1=Night)': 'earlybird_nightowl',
                        'Extroversion (1-5)': 'extroversion',
                        'Good Advice (1-5)': 'good_advice',
                        'Plans Style (1-5)': 'plans_style',
                        'Study Frequency (1-5)': 'study_frequency',
                        'Gym Frequency (1-5)': 'gym_frequency',
                        'Spending Habits (1-5)': 'spending_habits',
                        'Friday Night': 'friday_night',
                        'Additional Info (Optional)': 'additional_info'
}

DEFAULT_IDENTIFIER = ['role',
                       'name',
                        'major', 
                        'year',
                        'ufl_email']
DEFAULT_PROFILE_TEXT = [
                        'free_time',
                        'hobbies',
                        'self_description',
                        'dislikes',
                        'talk_for_hours_about',
                        'friday_night',
                        'additional_info'
]

DEFAULT_CATEGORICALS = [
                        'major',
                        'earlybird_nightowl',
]

DEFAULT_NUMERICS     = [
                        'extroversion',
                        'good_advice',
                        'plans_style',
                        'study_frequency',
                        'gym_frequency',
                        'spending_habits'
]
