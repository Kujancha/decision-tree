import pandas as pd

data = {
    'weight': [220, 180, 250, 190, 300, 160, 270, 200, 240, 170],
    'height': [90, 85, 100, 88, 110, 80, 105, 85, 95, 82],
    'tail_length': [100, 95, 105, 98, 120, 90, 110, 100, 108, 92],
    'body_length': [180, 170, 200, 175, 210, 160, 205, 172, 195, 165],
    'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'age': [5, 4, 6, 4, 7, 3, 6, 5, 5, 3]
}

df = pd.DataFrame(data)
df.to_csv("tiger_data.csv", index=False)
