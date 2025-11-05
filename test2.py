import pandas as pd
count = '/home/santiagouwu/.seisbench/datasets/stead/metadata.csv'

df = pd.read_csv(count)
trace_category = df['trace_category']

noise = trace_category[trace_category == 'noise'].count()
print(f'Noise samples: {noise}')
event = trace_category[trace_category == 'earthquake_local'].count()
print(f'Event samples: {event}')
