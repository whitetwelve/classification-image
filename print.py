from keras.models import load_model

# Gantilah 'data.h5' dengan nama file yang sesuai dengan model Anda
model = load_model('models/animal-image-classification.h5')

# Tampilkan ringkasan model
model.summary()