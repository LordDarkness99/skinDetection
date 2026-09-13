import h5py

model_path = 'model_kulit_terbaik.h5'

print("Mengecek isi struktur file .h5...")
with h5py.File(model_path, 'r') as f:
    print("Keys dalam file H5:", list(f.keys()))
    if 'model_weights' in f:
        print("Ditemukan grup model_weights.")
        weights_group = f['model_weights']
        print("Sub-keys di model_weights:", list(weights_group.keys())[:10])