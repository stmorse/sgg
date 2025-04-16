import os

import numpy as np
import pandas as pd

BASE_PATH = '/sciclone/geograd/stmorse/reddit/subreddit/science/users'

# set date range
sy, ey = 2007, 2010
years = [y for y in range(sy, ey + 1)]

for year in years:
    print(f'Loading year {year} ...')

    # Load the user counts
    csv_file_path = os.path.join(BASE_PATH, f'user_label_counts_{year}.csv')
    df = pd.read_csv(csv_file_path)

    # Convert to ndarray
    data = df.drop(columns=['author']).values

    # Normalize using L2 norm
    l2_norms = np.linalg.norm(data, ord=2, axis=1, keepdims=True)
    normalized_data = data / l2_norms

    # Save the resulting ndarray as an NPZ file
    output_file_path = os.path.join(
        BASE_PATH, f'user_label_counts_L2norm_{year}.npz')
    np.savez(output_file_path, data=normalized_data)

    print(f"> Normalized data saved to {output_file_path}")