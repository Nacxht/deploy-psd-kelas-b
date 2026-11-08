import os
import urllib.request
import pandas as pd

bukan_air_dir = "C:/Users/Raihan aryanova/Desktop/proyek_hemat_air/dataset/bukan_air"
os.makedirs(bukan_air_dir, exist_ok=True)

print("[1/2] Mengunduh suara lingkungan/masjid (langkah kaki, batuk, tertawa, tepukan) dari ESC-50...")
esc_meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
df_esc = pd.read_csv(esc_meta_url)
noise_cats = ["clapping", "footsteps", "laughing", "coughing", "door_wood_knock"]
esc_samples = df_esc[df_esc["category"].isin(noise_cats)].head(35)

count = 0
for idx, row in esc_samples.iterrows():
    filename = row["filename"]
    cat = row["category"]
    url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
    dest = os.path.join(bukan_air_dir, f"ambient_{count+1:03d}_{cat}.wav")
    try:
        urllib.request.urlretrieve(url, dest)
        count += 1
    except Exception as e:
        print(f"Error {filename}: {e}")

print(f"-> Berhasil mengunduh {count} file ambient!")

print("[2/2] Mengunduh suara orang bicara/ngobrol (Human Speech) dari CMU Arctic...")
speech_count = 0
for i in range(1, 36):
    arctic_id = f"arctic_a{i:04d}"
    speaker = "cmu_us_awb_arctic" if i % 2 == 0 else "cmu_us_bdl_arctic"
    url = f"http://festvox.org/cmu_arctic/cmu_arctic/{speaker}/wav/{arctic_id}.wav"
    dest = os.path.join(bukan_air_dir, f"speech_{speech_count+1:03d}.wav")
    try:
        urllib.request.urlretrieve(url, dest)
        speech_count += 1
        count += 1
    except Exception as e:
        print(f"Error speech {arctic_id}: {e}")

print(f"-> Berhasil mengunduh {speech_count} file orang bicara!")
print(f"\n[SELESAI] Total dataset kebisingan/ngobrol terkumpul: {count} file di dataset/bukan_air!")
