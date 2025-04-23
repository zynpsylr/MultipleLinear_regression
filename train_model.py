import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib   # Modeli kaydetmek için kullanılır

# Veriyi oku
data = pd.read_csv('Advertising Budget and Sales.csv')

# Değişkenleri ayır
x= data[['TV Ad Budget ($)','Radio Ad Budget ($)','Newspaper Ad Budget ($)']]
y= data['Sales ($)']

# Eğitim ve test seti
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Verilerin ölçeklendirilmesi
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)


print("Model başarıyla eğitildi ve multiplelinear_model.pkl olarak kaydedildi.")
print("R2 Skoru         : ", r2_score(y_test, y_pred))                            # Modelin genel doğruluğu
print("Mean Squared Error (MSE): ", mean_squared_error(y_test, y_pred))           # Hataların karesel ortalaması
print("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, y_pred))         # Hataların mutlak ortalaması


# Modeli ve scaler'ı kaydet
joblib.dump(model, 'multiplelinear_model.pkl')  
joblib.dump(scaler, 'scaler.pkl')               
