

result = get_features('') #path predict
result = pd.DataFrame(result)


scaler = StandardScaler()
result = scaler.fit_transform(result)
result = np.expand_dims(result, axis=2)

pred_test = model.predict(result)
y_pred = encoder.inverse_transform(pred_test)

# df = pd.DataFrame(columns=['predicted', 'actual'])
# df['predicted'] = y_pred.flatten()
# df['actual'] = 'happy'
# df.head(5) 

print(y_pred)