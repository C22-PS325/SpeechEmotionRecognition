import pandas as pd
import numpy as np

result = get_features('') #path predict
result = pd.DataFrame(result)


scaler = StandardScaler()
result = scaler.fit_transform(result)
result = np.expand_dims(result, axis=2)

pred_test = model.predict(result)
y_pred = encoder.inverse_transform(pred_test)
